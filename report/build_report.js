// Builds report/report.docx following the college AAT template.
// Every number is read from results/*.json (never typed by hand).
//   node report/build_report.js   then   powershell report/export_pdf.ps1  (updates TOC, exports PDF)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, BorderStyle, ShadingType, TableOfContents, Footer, PageNumber,
  TabStopType, LevelFormat, PageBreak, VerticalAlign,
} = require(process.env.DOCX_MODULE || 'docx');

const ROOT = path.resolve(__dirname, '..');
const R = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, 'results', p), 'utf8'));
const team = JSON.parse(fs.readFileSync(path.join(__dirname, 'team.json'), 'utf8'));
const S = R('data_stats.json'), Q = R('data_quality.json'), P = R('metrics_pretrained.json');
const FC = R('final_config.json'), M = R(`runs/${FC.final_run}.json`), CV = R('metrics_cv.json');
const AB = R('metrics_ablations.json'), EF = R('metrics_efficiency.json'), EA = R('error_analysis.json');
const QE = R('qualitative_examples.json'), BA = R('metrics_before_after.json');
const ENV = Object.fromEntries(fs.readFileSync(path.join(ROOT, 'results', 'environment.txt'), 'utf8')
  .split('\n').filter((l) => l.includes('==')).map((l) => l.trim().split('==')));
const CVRUN = R('runs/cv_main.json');

const pct = (v) => (100 * v).toFixed(1);
const n = (v) => v.toLocaleString('en-US');
const TYPES = ['NAME', 'EMAIL', 'LOCATION', 'COLLEGE_NAME', 'DEGREE', 'GRADUATION_YEAR',
  'COMPANIES_WORKED_AT', 'DESIGNATION', 'YEARS_OF_EXPERIENCE', 'SKILLS'];
const before = BA.views.all_types.before, after = BA.views.all_types.after, afterR = BA.views.all_types.after_rules;
const ov = BA.views.overlap_types;
const ftCV = CVRUN.summary;
const grid = FC.grid;
const OS = R('runs/abl_os.json');
const valLossMin = M.history.reduce((b, e) => (e.val_loss < b.val_loss ? e : b)).epoch;
const ablBy = (prefix) => AB.ablations.find((a) => a.name.startsWith(prefix));
const sortedF1 = [...TYPES].sort((a, b) => after.per_type[a].f1 - after.per_type[b].f1);

// ---------------------------------------------------------------- styling helpers
const FONT = 'Times New Roman';
const NAVY = '1F3864', RED = 'FF0000', PURPLE = '7030A0';
const PAGE_W = 11906, MARGIN = 1440, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 1" margins
const border = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const borders = { top: border, bottom: border, left: border, right: border };

const figs = [], tabs = [];
const figNo = (key) => { if (!figs.includes(key)) figs.push(key); return figs.indexOf(key) + 1; };
const tabNo = (key) => { if (!tabs.includes(key)) tabs.push(key); return tabs.indexOf(key) + 1; };
// pre-register order so in-text references can precede the object
['system_architecture', 'demo_compare', 'demo_fields', 'bert_ner_architecture', 'label_distribution',
  'experiment_workflow', 'training_curves', 'before_after_f1', 'confusion_matrix', 'cv_folds', 'ablations',
].forEach(figNo);
const lr = (v) => Number(v).toExponential(0); // 3e-5
['entity_types', 'parsers', 'api_io', 'dataset', 'entity_types_stats', 'alignment', 'split', 'tools', 'hyper', 'grid',
  'before_after', 'overall', 'cv', 'efficiency', 'ablations', 'qualitative', 'errors', 'comparison'].forEach(tabNo);
const F = (k) => `Figure ${figNo(k)}`, T = (k) => `Table ${tabNo(k)}`;

// parse **bold** and *italic* markers inside a string
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const p = (text, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, ...opts, children: runs(text),
});
let listInstance = 0;
const bullets = (items, ref = 'bullets') => { const inst = ++listInstance; return items.map((t) => new Paragraph({
  numbering: { reference: ref, level: 0, instance: inst }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: 288 }, children: runs(t),
})); };
const numbered = (items) => bullets(items, 'numbers');
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function figure(key, file, caption, widthIn = 6.2) {
  const img = pngSize(path.join(ROOT, file));
  const wpx = widthIn * 96, hpx = (wpx * img.h) / img.w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: img.data, transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: key } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `${F(key)}: `, bold: true, size: 20 }), ...runs(caption, { size: 20 })] }),
  ];
}
function table(key, caption, header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const fs_ = opts.size || 19;
  const cell = (text, i, isHead, rowOpts = {}) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: isHead ? { fill: 'D9E2F3', type: ShadingType.CLEAR, color: 'auto' }
      : rowOpts.fill ? { fill: rowOpts.fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: String(text).split('\n').map((line) => new Paragraph({
      alignment: (opts.align && opts.align[i]) || AlignmentType.LEFT,
      children: runs(line, { size: fs_, bold: isHead || rowOpts.bold }),
    })),
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `${T(key)}: `, bold: true, size: 20 }), ...runs(caption, { size: 20 })] }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => {
          const ro = r.__opts || {};
          const vals = r.__opts ? r.vals : r;
          return new TableRow({ cantSplit: true, children: vals.map((v, i) => cell(v, i, false, ro)) });
        }),
      ],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const C = AlignmentType.CENTER;
const eq = (text, num) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
  children: [new TextRun({ text, font: 'Cambria Math', size: 23 }), new TextRun({ text: `\t(${num})`, size: 22 })],
});
const code = (lines) => lines.map((l, i) => new Paragraph({
  shading: { fill: 'F2F2F2', type: ShadingType.CLEAR, color: 'auto' },
  spacing: { before: i === 0 ? 120 : 0, after: i === lines.length - 1 ? 160 : 0, line: 260 },
  indent: { left: 200, right: 200 },
  children: [new TextRun({ text: l.replace(/ /g, ' '), font: 'Consolas', size: 17 })],
}));

// ---------------------------------------------------------------- cover page
const centered = (text, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 0 },
  children: [new TextRun({ text, font: FONT, ...o })] });
const logo = pngSize(path.join(__dirname, 'assets', 'bmsce_logo.png'));
const coverCell = (paras, w) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
  margins: { top: 100, bottom: 100, left: 120, right: 120 }, verticalAlign: VerticalAlign.CENTER, children: paras });
const line = (t, o = {}) => new Paragraph({ indent: { left: o.indent || 0 }, spacing: { after: 20 },
  children: [new TextRun({ text: t, size: 24, ...o })] });
const [m1, m2] = team.members;
const cover = [
  centered('B. M. S. COLLEGE OF ENGINEERING', { bold: true, size: 32, color: NAVY }),
  centered('Autonomous Institute, Affiliated to VTU, Belagavi', { italics: true, size: 20, after: 280 }),
  centered('DEPARTMENT OF MACHINE LEARNING', { bold: true, size: 32, color: NAVY }),
  centered('Program: B.E. In Artificial Intelligence and Machine Learning', { size: 26 }),
  centered('Academic Year: 2026 - 27, Session: July 2026 - Dec 2026', { bold: true, size: 20, after: 120 }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new ImageRun({ type: 'png',
    data: logo.data, transformation: { width: 66, height: (66 * logo.h) / logo.w },
    altText: { title: 'BMSCE logo', description: 'B. M. S. College of Engineering logo', name: 'logo' } })] }),
  centered('Natural Language Processing for LLM', { bold: true, size: 34, color: RED }),
  centered(`(${team.course_code})`, { bold: true, size: 34, color: RED }),
  centered('Alternative Assessment Tool (AAT) Report', { bold: true, size: 24, after: 200 }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
    new TextRun({ text: 'Topic: ', bold: true, size: 28, color: PURPLE, font: FONT }),
    new TextRun({ text: team.topic, bold: true, size: 26, color: PURPLE, underline: {}, font: FONT })] }),
  centered('Submitted by', { bold: true, size: 24, after: 120 }),
  new Table({ width: { size: 7300, type: WidthType.DXA }, columnWidths: [3300, 4000], alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ children: [
        coverCell([line('Student Name:'), line('Team Member 1:', { indent: 500 }), line('Team Member 2:', { indent: 500 })], 3300),
        coverCell([line(' '), line(m1.name), line(m2.name)], 4000)] }),
      new TableRow({ children: [
        coverCell([line('USN:'), line('Team Member 1:', { indent: 500 }), line('Team Member 2:', { indent: 500 })], 3300),
        coverCell([line(' '), line(m1.usn), line(m2.usn)], 4000)] }),
      new TableRow({ children: [
        coverCell([line('Semester & Section:', { indent: 300 })], 3300),
        coverCell([line(`${team.semester}th Semester, Section ${team.section}`)], 4000)] }),
    ] }),
  new Paragraph({ spacing: { before: 600 }, children: [] }),
  centered('Valuation Report', { bold: true, size: 24, color: RED }),
  centered('(to be filled by faculty)', { italics: true, size: 22, after: 60 }),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [3600, CONTENT_W - 3600],
    rows: [['Score:', ''], ['Comments:', ''], ['Faculty In-charge:', 'Upeksha Prasad G'], ['Faculty Signature with date:', '']]
      .map(([a, b]) => new TableRow({ height: { value: 560, rule: 'atLeast' }, children: [
        coverCell([line(a, { bold: true })], 3600), coverCell([line(b)], CONTENT_W - 3600)] })) }),
];

// ---------------------------------------------------------------- content
const secs = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? secs.push(...x) : secs.push(x)));

const toc = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, pageBreakBefore: true,
    children: [new TextRun({ text: 'Table of Contents', bold: true, size: 36, font: FONT })] }),
  new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-2' }),
];

// ===== 1
const bestVal = FC.best;
add(h1('1. Introduction'));
add(h2('1.1 Background of Natural Language Processing'));
add(p('Natural Language Processing (NLP) studies how computers can analyse, understand and generate human language. Early systems relied on hand-written rules and lexicons; statistical NLP then introduced probabilistic sequence models such as Hidden Markov Models and Conditional Random Fields, which learned from annotated corpora [1]. The neural era replaced hand-crafted features with learned representations: recurrent networks and BiLSTM-CRF taggers [12] became the standard for sequence labelling, and the Transformer architecture [2] removed recurrence entirely in favour of self-attention. Pretrained Transformer encoders such as BERT [3] are trained once on large unlabelled corpora and then *fine-tuned* on a specific task with a small labelled dataset. This "pretrain, then fine-tune" paradigm underlies today\'s large language models (LLMs) and is the central idea examined in this report.'));
add(h2('1.2 Importance of the Selected NLP Topic'));
add(p('Recruiters and Applicant Tracking Systems (ATS) receive large volumes of resumes in heterogeneous formats (PDF, DOCX, plain text) with no fixed structure. Before candidates can be searched, filtered or matched against a job description, the unstructured resume text has to be turned into structured fields: name, contact details, education, employers, job titles and skills. Named Entity Recognition (NER) is the NLP task that performs exactly this extraction. Commercial resume-parsing services expose this capability through REST APIs that return structured JSON [18], [19], which shows the practical value of the problem.'));
add(p(`An off-the-shelf NER model, however, is trained on news text and only knows generic entity types (person, organisation, location). Resumes use a different vocabulary, a different layout and need domain-specific labels such as DEGREE, DESIGNATION or SKILLS. Measuring how much *fine-tuning* closes this gap is therefore both a practical and an instructive question for a course on NLP for LLMs.`));
add(h2('1.3 Objective of the Study'));
add(...numbered([
  'Build a resume-parsing system based on NER that converts raw resume text (PDF, DOCX or TXT) into a list of typed entities with character spans and a structured JSON record.',
  `Measure the **normal performance** of an existing pretrained NER model, *dslim/bert-base-NER* [5], applied to resumes without any training (the "before" condition).`,
  'Fine-tune the same checkpoint on an annotated resume dataset and measure its performance on the same test data with the same metric (the "after" condition).',
  'Study performance-enhancement techniques (learning rate and epochs, class-weighted loss, oversampling, sliding windows, a regex rule layer and text normalisation) as an ablation.',
  'Deploy both models behind a REST API and a web demo that shows the before-versus-after difference live.',
]));
add(h2('1.4 Scope of the Report'));
add(p(`The study uses the publicly available DataTurks "Resume Entities for NER" dataset [9] (${S.n_resumes} unique English resumes, ${TYPES.length} entity types). One model family is studied in two conditions (pretrained and fine-tuned BERT-base); larger LLMs and layout-aware models are discussed only as future work. All numbers in this report were produced by the project scripts and are stored in the results folder of the code repository; nothing has been estimated or copied from other work. Training was performed on a single laptop GPU (${M.config.device}) and inference is evaluated on a laptop CPU, reflecting a realistic small-team deployment.`));

// ===== 2
add(h1('2. Selected NLP Topic Overview'));
add(h2('2.1 Description of the NLP Task'));
add(p('Named Entity Recognition is a *sequence-labelling* task: given a sequence of words w₁ … wₙ, the model assigns each word a label that marks whether it begins an entity, continues an entity, or lies outside any entity. In this project the input is raw resume text and the output is a list of entities, each with a type, the surface text and a character span (start, end).'));
add(p(`Labels follow the **BIO (IOB2) scheme** [10]: B-X marks the first word of an entity of type X, I-X marks a following word of the same entity and O marks all other words. With ${TYPES.length} entity types this gives 2 × ${TYPES.length} + 1 = ${M.config.labels.length} labels. ${T('entity_types')} lists the entity types used. For example, the header line "Ananya Rao / Data Scientist" is tagged *Ananya/B-NAME Rao/I-NAME Data/B-DESIGNATION Scientist/I-DESIGNATION*.`));
add(...table('entity_types', 'Target entity types (all present in the DataTurks annotations)',
  ['Entity type', 'Meaning', 'Example (from the data)'],
  [['NAME', 'Candidate name', 'Ashok Kunam'], ['EMAIL', 'Contact link (in this dataset always an Indeed profile URL)', 'indeed.com/r/Ashok-Kunam/…'],
    ['LOCATION', 'City / region of the candidate', 'Bengaluru'], ['COLLEGE_NAME', 'School, college or university', 'Visvesvaraya Technological University'],
    ['DEGREE', 'Degree or qualification', 'Bachelor of Engineering in Engineering'], ['GRADUATION_YEAR', 'Year a degree was completed', '2017'],
    ['COMPANIES_WORKED_AT', 'Employer names', 'Microsoft'], ['DESIGNATION', 'Job title', 'Server Support Engineer'],
    ['YEARS_OF_EXPERIENCE', 'Stated total experience', '2.5 years'], ['SKILLS', 'Skill lists or skills sections', 'Python, SQL, …']],
  [2400, 3700, 2926]));
add(h2('2.2 Real-World Use Cases'));
add(...bullets([
  '**Applicant Tracking Systems and recruitment platforms:** automatic population of candidate profiles from uploaded resumes, so recruiters can search by skill, employer or degree.',
  '**Candidate–job matching:** extracted skills and experience are matched against job descriptions to rank applicants.',
  '**HR analytics:** aggregating degrees, employers or skills across thousands of applications.',
  '**Commercial parsing APIs:** services such as Affinda and Textkernel offer resume parsing as a paid API that returns structured JSON [18], [19].',
  '**Beyond hiring:** the same NER technique is used for invoices, medical notes, legal contracts and news (people, organisations, places) [1].',
]));
add(h2('2.3 Relevance to the Syllabus and Industry'));
add(p('The project covers core syllabus topics of NLP for LLMs: tokenisation (word-level and WordPiece sub-words), sequence labelling, contextual embeddings, the Transformer encoder and self-attention, transfer learning through pretraining and fine-tuning, and evaluation with precision, recall and F1. It also illustrates the industrial workflow of adapting a public pretrained checkpoint from the Hugging Face hub [6] to a domain-specific task and serving it through an API, which is how most production NLP systems are built today.'));
add(h2('2.4 Key Challenges in This NLP Task'));
add(...bullets([
  '**Domain shift:** the pretrained model was trained on Reuters news (CoNLL-2003 [4]); resumes have short, list-like lines, abbreviations and Indian names and institutions that are rare in news.',
  '**Label-set mismatch:** CoNLL-2003 has only PER, ORG, LOC and MISC; seven of our ten target types (e.g. DEGREE, SKILLS, DESIGNATION) have no counterpart at all.',
  '**Ambiguity:** the same string can be a company, a college or a skill ("Oracle" as employer vs. skill); job titles and company names appear side by side on one line.',
  '**Layout noise:** text extracted from PDFs loses columns, tables and line structure; bullets turn into replacement characters.',
  `**Small, noisy data:** only ${S.n_resumes} annotated resumes are available, with inconsistent annotation (Section 8.4), and some entity types are rare (YEARS_OF_EXPERIENCE: ${S.entities_per_type.YEARS_OF_EXPERIENCE} spans in total).`,
  `**Long documents:** resumes average ${Math.round(S.tokens_per_resume.mean)} words (maximum ${n(S.tokens_per_resume.max)}), exceeding BERT\'s 512 sub-token limit.`,
]));

// ===== 3
add(h1('3. Real-Time Application / API Identified'));
add(h2('3.1 Application or API Name'));
add(p('**Resume NER API and demo** (developed in this project): a FastAPI service with the endpoint *POST /parse* and a React web interface. The service wraps both models studied in this report, so the same resume can be parsed by the pretrained model and by the fine-tuned model and compared side by side.'));
add(p(`For real-world context, ${T('parsers')} compares our API with two commercial resume-parsing APIs, described only from their public documentation.`));
add(...table('parsers', 'Our API compared with commercial resume-parsing APIs (public documentation only)',
  ['Aspect', 'This project', 'Affinda Resume Parser [18]', 'Textkernel Tx Platform [19]'],
  [['Access', 'Local FastAPI service, POST /parse (multipart file) and POST /parse_text', 'REST API; v3 upload to /v3/documents with API key, workspace and document-type identifiers', 'SaaS REST API; Base64-encoded document sent with account ID and service key headers'],
    ['Output', 'JSON: name, email, phone, skills, education, companies, designations, entities with spans', 'Structured JSON for the parsed resume', 'JSON following a published schema'],
    ['Notes', 'Two models selectable (pretrained / fine-tuned); open source; runs on a laptop CPU', 'Documented rate limit and credit-based usage', 'Asks the caller to supply the document revision date for correct date interpretation']],
  [1500, 2600, 2463, 2463], { size: 17 }));
add(h2('3.2 Purpose and Functionality'));
add(p('The application accepts a resume file and returns (i) every detected entity with its label and character offsets, and (ii) a structured record that groups entities into the fields an ATS needs. In the web demo the user can upload a file, pick one of the sample resumes, or paste text, and then view the result of the pretrained model, the fine-tuned model, or both side by side. For resumes taken from the held-out test split the gold annotation is shown as a third column together with a per-document F1 score, which makes the effect of fine-tuning directly visible.'));
add(h2('3.3 Architecture or Platform on Which It Is Built'));
add(p(`${F('system_architecture')} shows the processing pipeline, which is shared by the command-line tool, the API and the demo. Text is extracted with *pdfplumber* (PDF) or *python-docx* (DOCX, including tables), normalised without changing character offsets, split into words with a regular expression, converted into WordPiece sub-tokens and processed in overlapping windows of 512 sub-tokens with a stride of 128. The BERT encoder and token-classification head produce a label distribution for every sub-token; predictions are taken from the first sub-token of each word and averaged where windows overlap. Finally the label sequence is repaired to valid BIO, an optional regex layer is applied, and the spans are grouped into JSON. The backend is Python (FastAPI, PyTorch, Hugging Face Transformers [6]); the frontend is React built with Vite and served by the same FastAPI process.`));
add(...figure('system_architecture', 'results/figures/system_architecture.png', 'End-to-end architecture of the resume-parsing application', 6.4));
add(h2('3.4 Input/Output Behavior'));
add(...table('api_io', 'API input and output', ['Item', 'Description'],
  [['Input', 'multipart/form-data: file (PDF, DOCX or TXT, ≤ 10 MB), model = pretrained | finetuned | both, rules = true | false'],
    ['Output fields', 'name, email, phone, location[], skills[], education{degrees[], colleges[], graduation_years[]}, companies[], designations[], years_of_experience[]'],
    ['entities[]', 'list of {text, label, start, end} with character offsets into the extracted text'],
    ['Diagnostics', 'model, rules, n_words, latency_ms (model inference only, model load excluded)'],
    ['Errors', '422 for unsupported file types, empty text or image-only PDFs; 413 for files over 10 MB; 503 if the fine-tuned model is missing']],
  [2000, 7026]));
add(p(`Measured on the laptop CPU (${EF.threads} threads) over the ${EF.n_resumes} test resumes (average ${Math.round(EF.mean_words_per_resume)} words), inference takes ${EF.finetuned.sec_per_resume_mean.toFixed(2)} s per resume on average for the fine-tuned model (median ${EF.finetuned.sec_per_resume_median.toFixed(2)} s, 95th percentile ${EF.finetuned.sec_per_resume_p95.toFixed(2)} s) and ${EF.pretrained.sec_per_resume_mean.toFixed(2)} s for the pretrained model. The two models have the same architecture, so the cost is essentially identical; latency grows linearly with the number of 512-token windows. ${F('demo_compare')} and ${F('demo_fields')} show the web demo.`));
add(...figure('demo_compare', 'results/screenshots/demo_compare.png', 'Web demo: the same test resume with gold annotation, pretrained output and fine-tuned output (per-document F1 shown in the panel headers)', 5.3));
add(...figure('demo_fields', 'results/screenshots/demo_fields.png', 'Web demo, structured-field view for a fictional sample resume uploaded as PDF', 6.2));
add(h2('3.5 Limitations and Practical Considerations'));
add(...bullets([
  '**Scanned PDFs:** image-only PDFs contain no text layer; the API returns an error because OCR is not part of the pipeline.',
  '**Multi-column layouts and tables:** pdfplumber reads text line by line, so two-column resumes can interleave the columns; DOCX tables are flattened to "cell | cell" lines.',
  `**Long resumes:** latency grows with length (95th percentile ${EF.finetuned.sec_per_resume_p95.toFixed(1)} s on CPU); a GPU or a smaller distilled model would be needed for high throughput.`,
  `**Model size:** the fine-tuned model is ${Math.round(EF.finetuned.model_size_mb)} MB (${(EF.finetuned.n_parameters / 1e6).toFixed(1)} M parameters), which must be loaded into memory once at start-up.`,
  '**Accuracy:** with entity-level F1 below 50 % on several types (Section 7), the output should support, not replace, a human recruiter.',
  '**Privacy and fairness:** resumes contain personal data; a production system must store as little as possible and should be audited for bias before influencing hiring decisions [16].',
]));

// ===== 4
add(h1('4. NLP Model / Architecture Study'));
add(h2('4.1 Theoretical Foundations of the Model'));
add(p('The Transformer [2] represents every token by a vector and repeatedly updates these vectors with *self-attention*: each token builds a query, compares it with the keys of all other tokens, and takes a weighted average of their values. Because every token can attend to every other token in one step, the model captures long-range context in both directions, which is essential for NER where the type of a word often depends on its neighbours (e.g. "Infosys" after "System Engineer -").'));
add(p('BERT [3] (Bidirectional Encoder Representations from Transformers) is a stack of Transformer encoder layers pretrained on BooksCorpus and English Wikipedia with two self-supervised objectives: masked language modelling (predict randomly masked tokens from both left and right context) and next-sentence prediction. Pretraining gives contextual word representations that already encode syntax and much world knowledge. For a downstream task, a small task-specific layer is added on top and the whole network is *fine-tuned* end to end with a low learning rate.'));
add(h2('4.2 Algorithm(s) Used'));
add(p(`The model studied is **dslim/bert-base-NER** [5]. According to its model card it is the *bert-base-cased* checkpoint fine-tuned on the English CoNLL-2003 NER dataset [4] (Reuters news) with the labels PER, ORG, LOC and MISC in BIO format, trained on a single NVIDIA V100 GPU with the hyperparameters recommended in the BERT paper. The card reports a test F1 of 91.3 in its evaluation table (the metadata section lists 92.6). It has ${n(EF.pretrained.n_parameters)} parameters.`));
add(...bullets([
  `**Before (pretrained, normal performance):** the checkpoint is used exactly as published. Its labels are mapped to ours: PER→NAME, ORG→COMPANIES_WORKED_AT, LOC→LOCATION; MISC is discarded.`,
  `**After (fine-tuned):** the same checkpoint is loaded with a new ${M.config.labels.length}-label classification head (the old 9-label head is discarded, option *ignore_mismatched_sizes*), and all ${n(EF.finetuned.n_parameters)} parameters are trained on resumes with cross-entropy loss and the AdamW optimiser [13].`,
]));
add(h2('4.3 Architecture Diagram / Layer-Wise Design'));
add(p(`${F('bert_ner_architecture')} shows the layer-wise design. BERT-base has 12 encoder layers, hidden size 768, 12 attention heads of size 64 and a feed-forward inner size of 3072 with GELU activation [20]; each sub-layer is wrapped in a residual connection followed by layer normalisation [21]. The input embedding is the sum of a WordPiece token embedding (cased vocabulary of 28,996 tokens [11]), a learned position embedding (maximum 512 positions) and a segment embedding. The token-classification head is a single linear layer (with dropout) that maps each 768-dimensional hidden state to label scores. The only architectural difference between the two conditions is this head: 9 outputs before, ${M.config.labels.length} outputs after fine-tuning.`));
add(...figure('bert_ner_architecture', 'results/figures/bert_ner_architecture.png', 'BERT encoder with token-classification head; the real WordPiece segmentation of a resume line is shown at the bottom', 6.4));
add(h2('4.4 Data Flow in the Model'));
add(...numbered([
  'The resume text is split into words with character offsets, and each word is split into WordPiece sub-tokens (e.g. "Fintrix" → *Fin ##tri ##x*).',
  'The sub-token sequence is cut into windows of at most 512 tokens ([CLS] … [SEP]) that overlap by 128 tokens, so no word is lost on long resumes.',
  'Each window passes through the embedding layer and the 12 encoder layers, producing one contextual vector per sub-token.',
  'The classification head and a softmax give a probability distribution over labels for every sub-token.',
  'For each word, the distribution of its first sub-token is used; if a word appears in several windows, the distributions are averaged.',
  'The arg-max labels are repaired to a valid BIO sequence (an I-X without a preceding X becomes B-X) and decoded into character spans.',
]));
add(h2('4.5 Mathematical Formulation'));
add(p('For a window with hidden matrix H, each attention head computes queries, keys and values Q = HW_Q, K = HW_K, V = HW_V and the scaled dot-product attention [2]:'));
add(eq('Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V', 1));
add(p('where d_k = 64 is the head dimension. The outputs of the 12 heads are concatenated and projected. The classification head turns the final hidden state hᵢ of sub-token i into a label distribution:'));
add(eq('p(yᵢ = c | x) = softmax(W hᵢ + b)_c = exp(z_c) / Σ_c′ exp(z_c′),   z = W hᵢ + b', 2));
add(p(`with W ∈ ℝ^(${M.config.labels.length}×768). Training minimises the average cross-entropy over all labelled positions L (the first sub-token of every word; the remaining sub-tokens have label −100 and are ignored):`));
add(eq('ℒ = − (1/|L|) Σ_{i∈L} log p(yᵢ = gᵢ | x)', 3));
add(p('In the class-weighted ablation each term is multiplied by a weight α_g = (1/√freq(g)) normalised to mean 1. Evaluation uses entity-level precision, recall and F1, where an entity counts as correct only if both its boundaries and its type match exactly (seqeval strict mode [8]):'));
add(eq('P = TP / (TP + FP),   R = TP / (TP + FN),   F1 = 2PR / (P + R)', 4));
add(p('Micro-averaged scores pool TP, FP and FN over all entity types; macro-averaged scores are the unweighted mean of the per-type scores, so rare types count as much as frequent ones.'));
add(h2('4.6 Why This Model Suits the Selected NLP Task'));
add(...bullets([
  `Bidirectional context: the type of a resume entity depends on both neighbours (a year after a degree is a graduation year; a name next to "Engineer" is a company).`,
  `Already an NER model: dslim/bert-base-NER has learned what names, organisations and places look like, so fine-tuning starts from a strong initialisation rather than from scratch; this matters with only ${S.split_sizes.train} training resumes [3], [14].`,
  'Cased vocabulary: capitalisation is a strong cue for names and organisations in resumes.',
  'Token-level output fits BIO tagging directly; WordPiece handles unseen names and technical terms by decomposing them into sub-words instead of mapping them to an unknown token.',
  `Size: at about ${Math.round(EF.finetuned.n_parameters / 1e6)} M parameters, it can be fine-tuned on a 6 GB laptop GPU in minutes and served on a CPU in about ${EF.finetuned.sec_per_resume_mean.toFixed(1)} s per resume.`,
]));

// ===== 5
const al = S.alignment;
add(h1('5. Dataset and Preprocessing'));
add(h2('5.1 Dataset Description (Source, Size, Structure)'));
add(p(`The dataset is "Resume Entities for NER" published by DataTurks on Kaggle [9] (accessed 16 October 2026 with the Kaggle API; the licence is listed as "unknown" on Kaggle, so the data is used for academic purposes only). It is a single file in JSON-lines format: each of the ${S.n_resumes_raw} lines holds one resume as plain text ("content") and a list of annotations, each with a label and character offsets (start, inclusive end). The resumes were taken from Indeed profiles of Indian candidates. ${T('dataset')} summarises the data after removing ${S.n_duplicates_removed} exact duplicate, and ${T('entity_types_stats')} gives the entity statistics.`));
add(...table('dataset', 'Dataset summary', ['Property', 'Value'],
  [['Source', 'Kaggle: dataturks/resume-entities-for-ner [9]'], ['Raw resumes / after de-duplication', `${S.n_resumes_raw} / ${S.n_resumes}`],
    ['Word tokens (regex)', `${n(S.n_word_tokens)} (mean ${S.tokens_per_resume.mean.toFixed(1)}, min ${S.tokens_per_resume.min}, max ${n(S.tokens_per_resume.max)} per resume)`],
    ['Entity spans (after cleaning)', n(S.entities_total)], ['Tokens labelled O', `${pct(S.fraction_O_tokens)} %`],
    ['Entity types', `${TYPES.length} (+ 2 "UNKNOWN" annotations, dropped)`]],
  [3600, 5426]));
add(...table('entity_types_stats', 'Entity statistics per type (all 219 resumes)',
  ['Entity type', 'Spans', 'Resumes containing it', 'Mean length (words)', 'Max length (words)'],
  TYPES.slice().sort((a, b) => S.entities_per_type[b] - S.entities_per_type[a]).map((t) => [t, S.entities_per_type[t],
    Q.resumes_with_at_least_one_span_of_type[t], Q.span_length_words[t].mean.toFixed(1), Q.span_length_words[t].max]),
  [3000, 1300, 1900, 1500, 1326], { align: [null, C, C, C, C] }));
add(p(`The data is strongly imbalanced: ${pct(S.fraction_O_tokens)} % of all words are outside any entity, COMPANIES_WORKED_AT has ${S.entities_per_type.COMPANIES_WORKED_AT} spans while YEARS_OF_EXPERIENCE has only ${S.entities_per_type.YEARS_OF_EXPERIENCE}, and SKILLS spans are very long (mean ${Q.span_length_words.SKILLS.mean.toFixed(1)} words, up to ${Q.span_length_words.SKILLS.max}) because annotators often marked whole skill sections as one entity. ${F('label_distribution')} visualises the distribution.`));
add(...figure('label_distribution', 'results/figures/label_distribution.png', 'Label distribution: entity spans per type (left) and word tokens per type on a log scale (right)', 6.4));
add(h2('5.2 Data Cleaning Steps'));
add(...numbered([
  `**De-duplication:** resumes were hashed after whitespace normalisation and lower-casing; ${S.n_duplicates_removed} exact duplicate was removed. No identical text occurs in both training and test sets (checked: ${S.train_test_identical_texts}).`,
  '**Offset-preserving normalisation:** control characters, Unicode replacement characters (U+FFFD, produced by bullets during PDF conversion), private-use symbols and unusual space characters were replaced by a plain space, one character for one character, so all annotation offsets remain valid.',
  '**Span conversion:** DataTurks end offsets are inclusive, so 1 was added to obtain Python-style [start, end) spans. Leading and trailing whitespace was trimmed from every span; empty spans and the "UNKNOWN" label were dropped.',
  '**Overlap resolution:** some annotations overlap (e.g. a company inside a designation span). Spans were sorted by start position, longer first, and a span overlapping an already accepted span was dropped.',
  '**Word alignment:** spans whose boundaries fall inside a word (the offsets are sometimes shifted by a few characters) were expanded to whole-word boundaries.',
]));
add(...table('alignment', 'Span-alignment log produced by data_prep.py', ['Event', 'Count'],
  [['"UNKNOWN" labels dropped', al.dropped_unknown_label], ['Empty spans dropped', al.dropped_empty],
    ['Overlapping spans dropped', `${al.dropped_overlap} (${Object.entries(al.dropped_overlap_by_type).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${v}`).join(', ')}, …)`],
    ['Spans expanded to word boundaries', al.span_misaligned_expanded_to_word_boundary],
    ['Spans recovered exactly after BIO → span round trip', `${pct(al.roundtrip_recovered_pct / 100)} %`],
    ['Original (trimmed) spans recovered character-exactly', `${pct(al.original_span_exact_recovered_pct / 100)} %`]],
  [5800, 3226], { align: [null, C] }));
add(p(`As a check, the BIO tags were converted back to spans: ${al.roundtrip_recovered_pct.toFixed(1)} % of the aligned spans are recovered exactly. The remaining cases are spans whose shifted offsets, after expansion to word boundaries, share a word with a neighbouring entity (e.g. a DESIGNATION span that begins inside the heading word "EXPERIENCE").`));
add(h2('5.3 Tokenization, Lemmatization, Stopwords Removal'));
add(p('Two levels of tokenisation are used. First, text is split into **words** with the regular expression \\w+|[^\\w\\s], which keeps alphanumeric runs together and makes every punctuation mark its own token; the character offsets of every word are stored so that predictions can be mapped back to the original text. Second, each word is split into **WordPiece sub-tokens** by the BERT tokenizer [11] (passed with is_split_into_words=True), which preserves the word-to-sub-token alignment.'));
add(p('**Stopwords were not removed and no lemmatisation or lower-casing was applied**, deliberately. In NER every word must receive a label and entities themselves contain function words ("Bachelor *of* Engineering", "Officer-HR *&* Administration *in* H.& R. Johnson"); removing them would break entity boundaries and destroy the character alignment with the gold spans. Lemmatisation would alter surface forms that the model must copy exactly, and lower-casing would remove capitalisation, one of the strongest cues for names and organisations, which the cased BERT vocabulary relies on.'));
add(h2('5.4 Train/Test Split Strategy'));
add(p(`The ${S.n_resumes} resumes were split at document level (never splitting a resume) into **70 % training, 15 % validation and 15 % test** with a fixed random seed (42). The validation set is used only for model selection (best epoch and hyperparameters); the test set is used once for the final numbers. Because the test set is small, a **5-fold cross-validation** was also run: in each fold one fifth of the resumes forms the test part and the rest is split 85/15 into training and validation. The pretrained model needs no training, so it is simply evaluated on each fold\'s test part.`));
const se = S.split_entities;
add(...table('split', 'Resumes and entity spans per split', ['Split', 'Resumes', ...['NAME', 'COMP.', 'DESIG.', 'SKILLS', 'YRS EXP.'], 'All spans'],
  ['train', 'val', 'test'].map((k) => [k, S.split_sizes[k], se[k].NAME, se[k].COMPANIES_WORKED_AT, se[k].DESIGNATION, se[k].SKILLS,
    se[k].YEARS_OF_EXPERIENCE, Object.values(se[k]).reduce((a, b) => a + b, 0)]),
  [1200, 1100, 1000, 1100, 1100, 1000, 1200, 1326], { align: [null, C, C, C, C, C, C, C] }));
add(p(`The CV folds contain ${S.cv_fold_test_sizes.join(', ')} test resumes. Note that the test split contains only ${se.test.YEARS_OF_EXPERIENCE} YEARS_OF_EXPERIENCE spans, so scores for that type are very noisy.`));
add(h2('5.5 Feature Engineering or Embedding Approach'));
add(p('No hand-crafted features (TF-IDF, gazetteers, POS tags) are used. Each WordPiece sub-token is represented by BERT\'s input embedding (token + position + segment, 768 dimensions), and the 12 encoder layers turn these into **contextual embeddings**: the same word "Oracle" receives a different vector in "worked at Oracle" and in "Skills: Oracle, SQL". These contextual vectors are the features consumed by the classification head. In the fine-tuned model the embeddings themselves are also updated, so they adapt to resume vocabulary. Long resumes are handled by the sliding-window scheme of Section 4.4 rather than by truncation (compared in Section 7.3).'));

// ===== 6
add(h1('6. Implementation Methodology'));
add(h2('6.1 Tools, Libraries, and Environment Used'));
add(...table('tools', 'Software and hardware environment (from results/environment.txt)', ['Component', 'Version / details'],
  [['Python', '3.12.12'], ['PyTorch [17]', ENV.torch], ['Hugging Face Transformers [6]', ENV.transformers],
    ['seqeval [8]', ENV.seqeval], ['scikit-learn (splits)', ENV['scikit-learn']], ['pdfplumber / python-docx', `${ENV.pdfplumber} / ${ENV['python-docx']}`],
    ['FastAPI / Uvicorn', `${ENV.fastapi} / ${ENV.uvicorn}`], ['Frontend', 'React 19 + Vite'], ['matplotlib / seaborn', `${ENV.matplotlib} / ${ENV.seaborn}`],
    ['Training hardware', `${M.config.device} (6 GB), CUDA 13.0, bf16 mixed precision`], ['Inference hardware', `Laptop CPU (Intel, ${EF.threads} threads)`],
    ['Colab', 'notebooks/train_colab.ipynb runs the identical scripts on a free Colab GPU']],
  [3400, 5626]));
add(h2('6.2 Model Implementation Steps'));
add(h3('A. Before: pretrained, normal performance (pretrained_eval.py)'));
add(...numbered([
  'Load dslim/bert-base-NER and its tokenizer unchanged; no parameter is updated.',
  'Predict word-level CoNLL labels for every resume with the sliding-window inference of Section 4.4.',
  'Map labels to the resume label set (PER→NAME, ORG→COMPANIES_WORKED_AT, LOC→LOCATION, MISC→O) and repair the BIO sequence.',
  'Evaluate with seqeval in two views: all ten entity types, and only the three overlapping types. An alternative mapping lets an ORG entity count as COLLEGE_NAME when the gold type at its first word is COLLEGE_NAME (an optimistic upper bound).',
]));
add(h3('B. After: fine-tuned performance (finetune.py)'));
add(...numbered([
  `Load the same checkpoint with a new ${M.config.labels.length}-label head (id2label/label2id for our labels, ignore_mismatched_sizes=True).`,
  'Encode each training resume into overlapping 512-token windows; label the first sub-token of each word and set the other sub-tokens and special tokens to −100.',
  `Train with AdamW (weight decay ${M.config.weight_decay}, none on biases and LayerNorm), linear warm-up over ${pct(M.config.warmup)} % of steps followed by linear decay, gradient clipping at 1.0 and bf16 autocast.`,
  'After every epoch compute the validation loss and validation entity-level F1; keep the weights of the epoch with the best validation micro-F1.',
  'Evaluate the selected model on the same test split with the same metric as in A, with and without the regex rule layer.',
]));
add(h2('6.3 API Integration'));
add(p('parse_resume.py implements text extraction and inference once and is imported by the API. Both models are loaded lazily and cached, so model loading happens only on the first request. The FastAPI application exposes POST /parse (file upload), POST /parse_text (JSON), GET /samples, GET /metrics (the before-versus-after summary read from the results files) and serves the built React application and the result figures. Requesting model=both runs the two models on the same text, which is how the demo shows the comparison.'));
add(h2('6.4 Pseudocode / Workflow Diagram'));
add(p(`${F('experiment_workflow')} shows the experimental workflow; the core training and inference procedures are given below.`));
add(...figure('experiment_workflow', 'results/figures/experiment_workflow.png', 'Experimental design: one dataset, one split, two conditions, one metric', 6.4));
add(...code([
  'FINE-TUNE(train, val):',
  '  model ← load("dslim/bert-base-NER", num_labels=21, new head)',
  '  windows ← for each resume: WordPiece(words), cut into 512-token windows, stride 128',
  '  labels  ← first sub-token of word w gets BIO(w); other sub-tokens get −100',
  '  for epoch = 1 … E:',
  '      for batch in shuffle(windows):',
  '          loss ← CrossEntropy(model(batch), labels, ignore=−100)',
  '          AdamW step; linear warm-up/decay schedule; clip grad norm 1.0',
  '      f1 ← seqeval_F1(PREDICT(model, val), gold(val))',
  '      if f1 > best: best ← f1; keep weights',
  '  return model with best validation F1',
  '',
  'PREDICT(model, resume):',
  '  words, offsets ← regex_tokenize(clean(resume))',
  '  for each 512-token window: probs ← softmax(model(window))',
  '  P[w] ← mean over windows of probs at the first sub-token of word w',
  '  tags ← repair_BIO(argmax P);  [optional] tags ← regex_rules(tags)',
  '  return spans(tags, offsets)',
]));
add(h2('6.5 Hyperparameters and Training Configuration'));
add(p(`Learning rate and batch size were selected with a grid search (${T('grid')}) using **validation** micro-F1 only, so the test set did not influence any choice. The selected configuration is lr = ${lr(M.config.lr)}, batch size ${M.config.batch_size} (validation F1 ${pct(bestVal.best_val_micro_f1)} %). ${T('hyper')} lists the final configuration.`));
add(...table('hyper', 'Final training configuration', ['Hyperparameter', 'Value'],
  [['Base checkpoint', M.config.base_model], ['Learning rate', lr(M.config.lr)], ['Batch size', M.config.batch_size],
    ['Epochs (best epoch kept)', `${M.config.epochs} (best: ${M.best_epoch})`], ['Max length / stride', `${M.config.max_length} / ${M.config.stride} sub-tokens`],
    ['Weight decay', M.config.weight_decay], ['Warm-up', `${pct(M.config.warmup)} % of steps, then linear decay`], ['Optimiser', 'AdamW [13], gradient clipping 1.0'],
    ['Precision', M.config.precision], ['Seed', M.config.seed], ['Training windows / resumes', `${M.n_train_windows} / ${M.n_train_docs}`],
    ['Training time', `${Math.round(M.train_time_sec)} s on ${M.config.device}`]],
  [3600, 5426]));
add(...table('grid', 'Hyperparameter grid (10 epochs each, best epoch selected on validation)',
  ['Learning rate', 'Batch size', 'Best epoch', 'Val micro-F1 (%)', 'Test micro-F1 (%)'],
  grid.map((g) => ({ __opts: g.tag === bestVal.tag ? { bold: true, fill: 'E2EFDA' } : {},
    vals: [lr(g.lr), g.batch_size, g.best_epoch, pct(g.best_val_micro_f1), pct(g.test_micro_f1)] })),
  [1800, 1600, 1600, 2000, 2026], { align: [C, C, C, C, C] }));
add(p('The test column is reported for transparency only. Smaller batches (more update steps) were clearly better at every learning rate, and the selected configuration is not the one with the highest test score (lr 5e-5 scored marginally higher on test but lower on validation), which illustrates why selection must use the validation set.'));

// ===== 7
add(h1('7. Results and Evaluation'));
add(h2('7.1 Performance Metrics'));
add(p(`All scores are entity-level precision, recall and F1 computed with seqeval in strict IOB2 mode [8] on the held-out test split (${S.split_sizes.test} resumes, ${after.micro.support} gold entities). ${T('before_after')} gives the per-type comparison of the pretrained ("before") and fine-tuned ("after") model, and ${T('overall')} the overall scores.`));
add(...table('before_after', 'Per-type results on the test split: before (pretrained, mapped) vs after (fine-tuned). P/R/F1 in %',
  ['Entity type', 'Support', 'Before P', 'Before R', 'Before F1', 'After P', 'After R', 'After F1', 'Δ F1'],
  BA.rows.map((r) => [r.type, r.support, pct(r.before_p), pct(r.before_r), pct(r.before), pct(r.after_p), pct(r.after_r), `**${pct(r.after)}**`, `+${pct(r.delta)}`]),
  [2290, 800, 800, 800, 850, 800, 800, 850, 1036], { size: 17, align: [null, C, C, C, C, C, C, C, C] }));
const altAll = BA.pretrained_alt_mapping_test.all_types;
add(...table('overall', 'Overall scores on the test split (%)', ['Metric', 'Before', 'After', 'Δ'],
  [...BA.overall.map((o) => [o.metric, pct(o.before), `**${pct(o.after)}**`, `+${pct(o.delta)}`]),
    ['Micro P / R — all types', `${pct(before.micro.precision)} / ${pct(before.micro.recall)}`, `${pct(after.micro.precision)} / ${pct(after.micro.recall)}`, ''],
    ['Before with ORG→COMPANY or COLLEGE (alt. mapping), micro F1 all types', pct(altAll.f1), '—', ''],
    ['After + regex rule layer, micro F1 all types', '—', pct(afterR.micro.f1), '']],
  [5000, 1300, 1300, 1426], { align: [null, C, C, C] }));
add(p(`Fine-tuning raises micro-F1 over all ten types from ${pct(before.micro.f1)} % to ${pct(after.micro.f1)} % and macro-F1 from ${pct(before.macro.f1)} % to ${pct(after.macro.f1)} %. Even on the three types the pretrained model can express (NAME, COMPANIES_WORKED_AT, LOCATION), micro-F1 rises from ${pct(ov.before.micro.f1)} % to ${pct(ov.after.micro.f1)} %, so the gain is not only due to new labels. The alternative mapping, which lets ORG also count as COLLEGE_NAME, lifts the pretrained score only to ${pct(altAll.f1)} %.`));
add(...table('cv', '5-fold cross-validation, mean ± standard deviation over folds (%)', ['Model', 'Micro F1', 'Macro F1'],
  CV.rows.map((r) => [r.model, `${pct(r.micro_mean)} ± ${pct(r.micro_std)}`, `${pct(r.macro_mean)} ± ${pct(r.macro_std)}`]),
  [4600, 2200, 2226], { align: [null, C, C] }));
add(p(`Cross-validation confirms the result with lower variance than a single 33-resume test set: the fine-tuned model reaches ${pct(ftCV.test_micro_f1.mean)} ± ${pct(ftCV.test_micro_f1.std)} % micro-F1 against ${pct(P.cv_summary.all_types.micro_f1_mean)} ± ${pct(P.cv_summary.all_types.micro_f1_std)} % for the pretrained model; the fine-tuned model is better on every fold (${F('cv_folds')}). The single test split is somewhat easier than the CV average (${pct(after.micro.f1)} % vs ${pct(ftCV.test_micro_f1.mean)} %).`));
add(...table('efficiency', 'Efficiency (CPU inference on the test split, fastest of 3 repeats per resume)', ['Metric', 'Pretrained', 'Fine-tuned'],
  [['Parameters', n(EF.pretrained.n_parameters), n(EF.finetuned.n_parameters)],
    ['Mean / median latency per resume (s)', `${EF.pretrained.sec_per_resume_mean.toFixed(2)} / ${EF.pretrained.sec_per_resume_median.toFixed(2)}`, `${EF.finetuned.sec_per_resume_mean.toFixed(2)} / ${EF.finetuned.sec_per_resume_median.toFixed(2)}`],
    ['95th percentile latency (s)', EF.pretrained.sec_per_resume_p95.toFixed(2), EF.finetuned.sec_per_resume_p95.toFixed(2)],
    ['Throughput (words/s)', Math.round(EF.pretrained.words_per_sec), Math.round(EF.finetuned.words_per_sec)],
    ['Model size on disk', '—', `${Math.round(EF.finetuned.model_size_mb)} MB (safetensors)`],
    ['Training time', 'none', `${Math.round(EF.finetuned.train_time_sec)} s (GPU)`]],
  [4000, 2400, 2626], { align: [null, C, C] }));
add(h2('7.2 Visualization of Results'));
add(p(`${F('training_curves')} shows the training of the final model. Training loss falls steadily, while validation loss reaches its minimum at epoch ${valLossMin} and then rises slightly; validation F1, however, keeps improving until the last epoch (best epoch ${M.best_epoch}). The model therefore becomes more confident on some wrong predictions (higher loss) while still getting more entities exactly right; selecting the checkpoint by validation F1 rather than loss is the correct choice for an F1-evaluated task.`));
add(...figure('training_curves', 'results/figures/training_curves.png', 'Training and validation loss (left) and validation F1 per epoch (right) for the final model', 6.4));
add(p(`${F('before_after_f1')} compares F1 per entity type. The pretrained model scores only on NAME, LOCATION and COMPANIES_WORKED_AT and is zero by construction on the other seven types. ${F('confusion_matrix')} shows the word-level confusion matrix of the fine-tuned model (rows normalised).`));
add(...figure('before_after_f1', 'results/figures/before_after_f1.png', 'Entity-level F1 per type on the test split, before vs after fine-tuning', 6.4));
add(...figure('confusion_matrix', 'results/figures/confusion_matrix_finetuned.png', 'Word-level confusion matrix of the fine-tuned model on the test split (B/I collapsed, row-normalised)', 5.6));
const cm = M.confusion; const li = (l) => cm.labels.indexOf(l);
const rowN = (g, p_) => { const row = cm.matrix[li(g)]; return row[li(p_)] / row.reduce((a, b) => a + b, 0); };
add(p(`The dominant error is confusion with O (missed or unannotated words), e.g. ${pct(rowN('YEARS_OF_EXPERIENCE', 'O'))} % of YEARS_OF_EXPERIENCE words and ${pct(rowN('GRADUATION_YEAR', 'O'))} % of GRADUATION_YEAR words are predicted as O. Between entity types the largest confusions are LOCATION→EMAIL (${pct(rowN('LOCATION', 'EMAIL'))} %) and COLLEGE_NAME→DEGREE (${pct(rowN('COLLEGE_NAME', 'DEGREE'))} %). The LOCATION→EMAIL cell is largely an annotation artefact: in ${Q.indeed_url_spans_by_gold_type.LOCATION} resumes the annotators labelled the Indeed profile URL as LOCATION (Section 8.4).`));
add(...figure('cv_folds', 'results/figures/cv_folds.png', 'Micro-F1 on each cross-validation fold, before vs after', 4.6));
add(h2('7.3 Qualitative and Quantitative Analysis'));
add(h3('Quantitative: performance-enhancement ablations'));
add(p(`${T('ablations')} and ${F('ablations')} report each enhancement as a separate run that changes one setting of the final configuration; all rows use the same test split.`));
add(...table('ablations', 'Ablation study (%). Each row changes one setting of the final model', ['Setting', 'Val micro F1', 'Test micro F1', 'Test macro F1'],
  AB.ablations.map((a, i) => ({ __opts: i === 0 ? { bold: true, fill: 'E2EFDA' } : {},
    vals: [a.name, a.val_micro_f1 == null ? '—' : pct(a.val_micro_f1), pct(a.micro_f1), pct(a.macro_f1)] })),
  [4700, 1400, 1400, 1526], { align: [null, C, C, C] }));
const a = {
  rules: ablBy('+ regex'), ep5: ablBy('Epochs'), cw: ablBy('+ class'), os: ablBy('+ over'), tr: ablBy('Plain'), raw: ablBy('No text'),
};
add(...bullets([
  `**Learning rate and epochs:** training for 5 instead of 10 epochs drops test micro-F1 to ${pct(a.ep5.micro_f1)} %; the model is still under-trained after 5 epochs with this small learning rate. The learning-rate/batch-size grid is in ${T('grid')}.`,
  `**Class-weighted loss:** inverse-square-root class weights *hurt* (${pct(a.cw.micro_f1)} % micro, ${pct(a.cw.macro_f1)} % macro). Up-weighting rare I- labels makes the model over-predict long entities and lowers precision.`,
  `**Oversampling** resumes that contain rare types (${OS.oversampled_rare_types.join(', ')}) is roughly neutral: micro-F1 ${pct(a.os.micro_f1)} %, macro-F1 ${pct(a.os.macro_f1)} %.`,
  `**Sliding window vs truncation:** truncating each resume to its first 512 sub-tokens lowers test micro-F1 to ${pct(a.tr.micro_f1)} % and macro-F1 to ${pct(a.tr.macro_f1)} %; entities in the second half of long resumes (education, skills) are lost.`,
  `**Regex rule layer:** adds ${(100 * (a.rules.micro_f1 - after.micro.f1)).toFixed(1)} points micro-F1 (${pct(a.rules.micro_f1)} %). The gain comes entirely from EMAIL (${pct(after.per_type.EMAIL.f1)} → ${pct(afterR.per_type.EMAIL.f1)} %), where the regex recovers complete Indeed URLs; ${afterR.per_type.GRADUATION_YEAR.f1 === after.per_type.GRADUATION_YEAR.f1 ? 'the graduation-year rule did not change the score' : `GRADUATION_YEAR changes from ${pct(after.per_type.GRADUATION_YEAR.f1)} to ${pct(afterR.per_type.GRADUATION_YEAR.f1)} %`}.`,
  `**Text normalisation:** removing the offset-preserving clean-up gives ${pct(a.raw.micro_f1)} % vs ${pct(after.micro.f1)} %, i.e. no measurable effect; BERT\'s tokenizer already ignores most of the noise characters.`,
]));
add(...figure('ablations', 'results/figures/ablations.png', 'Test micro-F1 of each ablation', 5.8));
add(h3('Qualitative: the same snippets, both models'));
const pick = ['resume_009', 'resume_018', 'resume_045', 'resume_055'];
const fmtE = (es) => es.length ? es.map((e) => `${e.text.replace(/\s+/g, ' ')} → ${e.label}`).join('\n') : '—';
const qrows = pick.map((id) => QE.find((q) => q.doc === id)).filter(Boolean);
add(p(`${T('qualitative')} shows four of the ten test snippets that were analysed (all ten are in results/tables/qualitative.md). The pretrained model labels universities as companies, splits names, and tags state names as locations that the annotators did not mark; the fine-tuned model reproduces the resume-specific labels, including degrees and Indeed profile links.`));
add(...table('qualitative', 'Before vs after predictions on the same test snippets (entity → label)', ['Snippet (abridged)', 'Gold', 'Pretrained (before)', 'Fine-tuned (after)'],
  qrows.map((q) => [q.snippet.trim().split('\n').filter((l) => l.trim()).slice(0, 6).join('\n').slice(0, 260), fmtE(q.gold), fmtE(q.pretrained), fmtE(q.finetuned)]),
  [2500, 2176, 2175, 2175], { size: 15 }));
add(h2('7.4 Error Analysis'));
const ec = EA.categories;
const totalGold = ['correct', 'boundary', 'type', 'bound+type', 'missed'].reduce((s, k) => s + (ec[k] || 0), 0);
add(p(`Every gold entity of the test split was assigned to one error category by comparing it with the fine-tuned predictions (${T('errors')}): *correct* (exact span and type), *boundary* (overlapping prediction of the right type with different boundaries), *type* (exact span, wrong type), *boundary+type*, and *missed* (no overlapping prediction). Predictions that overlap no gold entity are *spurious*. Of ${totalGold} gold entities, ${ec.correct} are correct, ${ec.boundary} have boundary errors, ${(ec.type || 0) + (ec['bound+type'] || 0)} have type errors and ${ec.missed} are missed; in addition there are ${ec.spurious} spurious predictions.`));
add(...table('errors', 'Error categories of the fine-tuned model per entity type (test split, sorted by F1)', ['Entity type', 'F1 (%)', 'Correct', 'Boundary', 'Type', 'Bound.+type', 'Missed', 'Spurious'],
  [...EA.hardest_types_by_f1.map((h) => { const c = EA.per_type[h.type] || {};
    return [h.type, pct(h.f1), c.correct || 0, c.boundary || 0, c.type || 0, c['bound+type'] || 0, c.missed || 0, c.spurious || 0]; }),
  { __opts: { bold: true }, vals: ['Total', '', ec.correct, ec.boundary, ec.type || 0, ec['bound+type'] || 0, ec.missed, ec.spurious] }],
  [2500, 900, 950, 1000, 800, 1100, 850, 926], { size: 17, align: [null, C, C, C, C, C, C, C] }));
add(...bullets([
  `**SKILLS (F1 ${pct(after.per_type.SKILLS.f1)} %)** is the hardest type. Its errors are mostly *boundary* errors (${EA.per_type.SKILLS.boundary} of ${S.split_entities.test.SKILLS}): gold SKILLS spans are whole sections of up to ${Q.span_length_words.SKILLS.max} words whose exact start and end are arbitrary (sometimes including the heading "Technical Skills", sometimes not). Under strict matching, predicting the right section with slightly different boundaries counts as both a false positive and a false negative.`,
  `**YEARS_OF_EXPERIENCE (F1 ${pct(after.per_type.YEARS_OF_EXPERIENCE.f1)} %)** has only ${S.split_entities.train.YEARS_OF_EXPERIENCE} training examples and ${S.split_entities.test.YEARS_OF_EXPERIENCE} test examples; it is annotated in only ${Q.resumes_with_at_least_one_span_of_type.YEARS_OF_EXPERIENCE} of ${S.n_resumes} resumes, although phrases like "2.5 years of experience" occur far more often, so correct predictions are frequently counted as spurious.`,
  `**COLLEGE_NAME and DESIGNATION (≈ ${pct(after.per_type.COLLEGE_NAME.f1)} %)** suffer from boundary disagreements (e.g. whether the city after a university belongs to the name) and from the degree/college and designation/company ambiguity on header lines.`,
  `**Spurious predictions** are often real entities that were never annotated: e.g. "2.5 years" in the summary of resume_009 is predicted as YEARS_OF_EXPERIENCE but is not annotated. ${Q.resumes_with_url_not_annotated} resumes contain an Indeed URL that is not annotated as EMAIL, so a correct EMAIL prediction there is counted as an error.`,
]));
add(h2('7.5 Comparison Between Approaches'));
add(p(`${T('comparison')} summarises the comparison of the two approaches on identical data and metrics.`));
add(...table('comparison', 'Pretrained vs fine-tuned: summary', ['Criterion', 'Pretrained (before)', 'Fine-tuned (after)'],
  [['Label set', '4 CoNLL types; 3 mappable to resume types', 'All 10 resume types'],
    ['Test micro / macro F1 (all types)', `${pct(before.micro.f1)} / ${pct(before.macro.f1)} %`, `${pct(after.micro.f1)} / ${pct(after.macro.f1)} %`],
    ['Test micro F1 (NAME, COMPANY, LOCATION only)', `${pct(ov.before.micro.f1)} %`, `${pct(ov.after.micro.f1)} %`],
    ['5-fold CV micro F1', `${pct(P.cv_summary.all_types.micro_f1_mean)} ± ${pct(P.cv_summary.all_types.micro_f1_std)} %`, `${pct(ftCV.test_micro_f1.mean)} ± ${pct(ftCV.test_micro_f1.std)} %`],
    ['NAME F1', `${pct(before.per_type.NAME.f1)} %`, `${pct(after.per_type.NAME.f1)} %`],
    ['Precision / recall (micro, all types)', `${pct(before.micro.precision)} / ${pct(before.micro.recall)} %`, `${pct(after.micro.precision)} / ${pct(after.micro.recall)} %`],
    ['Training cost', 'none', `${Math.round(M.train_time_sec)} s on a laptop GPU`],
    ['CPU latency per resume', `${EF.pretrained.sec_per_resume_mean.toFixed(2)} s`, `${EF.finetuned.sec_per_resume_mean.toFixed(2)} s`]],
  [3600, 2700, 2726]));
add(p(`The pretrained model's low precision on its own types (${pct(before.per_type.COMPANIES_WORKED_AT.precision)} % for COMPANIES_WORKED_AT, ${pct(before.per_type.LOCATION.precision)} % for LOCATION) shows that it finds *many* organisations and places, but not the ones a resume parser wants: every university, skill vendor and state name is tagged, while the annotation marks only employers and the candidate's city.`));

// ===== 8
add(h1('8. Discussion'));
add(h2('8.1 Interpretation of Results'));
add(p(`Fine-tuning improves micro-F1 by ${pct(after.micro.f1 - before.micro.f1)} points on the test split and by ${pct(ftCV.test_micro_f1.mean - P.cv_summary.all_types.micro_f1_mean)} points in cross-validation. Two effects explain this. The first is the **label-set shift**: seven of the ten resume types simply do not exist in CoNLL-2003, so the pretrained model scores zero on them by construction. The second is the **domain and annotation-guideline shift**: even on the shared types, the pretrained model's notion of an entity differs from the resume annotation (all organisations vs only employers; any place vs the candidate's location; names split across header lines). Fine-tuning on ${S.split_sizes.train} resumes teaches both the new labels and the resume-specific guidelines, while reusing the language knowledge already in the encoder.`));
add(h2('8.2 Strengths and Weaknesses of the Model'));
add(...bullets([
  `**Strengths:** high accuracy on short, well-defined entities (NAME ${pct(after.per_type.NAME.f1)} %, LOCATION ${pct(after.per_type.LOCATION.f1)} %); robust to unseen names and technical terms thanks to WordPiece; handles arbitrarily long resumes through sliding windows; fast enough for interactive use on a CPU; one model covers all ten types.`,
  `**Weaknesses:** strict-match F1 stays below 40 % for long or ambiguous entities (${TYPES.filter((t) => after.per_type[t].f1 < 0.4).join(', ')}); the model sees text only, not layout; with a validation F1 of ${pct(M.best_val_micro_f1)} % and fold-to-fold standard deviation of ${pct(ftCV.test_micro_f1.std)} points, results are sensitive to the small amount of data.`,
]));
add(h2('8.3 Challenges Encountered During Implementation'));
add(...bullets([
  `**Span alignment:** DataTurks offsets are inclusive and sometimes shifted by a few characters; spans also overlap. A careful, logged conversion with a round-trip check was required (${T('alignment')}).`,
  '**Sub-word alignment:** labels exist per word but BERT works on sub-tokens; only the first sub-token is labelled and predicted, and predictions are merged back to words and characters.',
  `**Long resumes:** up to ${n(S.tokens_per_resume.max)} words; plain truncation loses ${pct(after.micro.f1 - a.tr.micro_f1)} points of micro-F1, so sliding windows with probability averaging were implemented for both training and inference.`,
  `**Tiny data:** ${S.split_sizes.train} training resumes make single-split results noisy; 5-fold CV and validation-only model selection were used to obtain trustworthy numbers.`,
  '**Engineering details:** Windows line endings in exported sample files shifted gold offsets in the demo, and Windows power throttling of background processes inflated CPU latency about five-fold; both were found during testing and fixed or controlled for (benchmarks are run in the foreground, fastest of three repeats).',
]));
add(h2('8.4 Impact of Dataset Quality and Preprocessing'));
add(p(`Annotation quality limits the achievable score. Measured on the full dataset: all ${Q.email_spans} EMAIL spans are Indeed profile URLs and none is an e-mail address; ${Q.resumes_with_url_not_annotated} resumes contain such a URL that was not annotated; in ${Q.indeed_url_spans_by_gold_type.LOCATION} resumes the URL is labelled LOCATION; ${Q.section_heading_but_no_annotation.COMPANIES_WORKED_AT.of_which_no_gold_span} of ${Q.section_heading_but_no_annotation.COMPANIES_WORKED_AT.resumes_with_heading} resumes with a WORK EXPERIENCE section have no company annotated and ${Q.section_heading_but_no_annotation.COLLEGE_NAME.of_which_no_gold_span} of ${Q.section_heading_but_no_annotation.COLLEGE_NAME.resumes_with_heading} resumes with an EDUCATION section have no college annotated. GRADUATION_YEAR is annotated in only ${Q.resumes_with_at_least_one_span_of_type.GRADUATION_YEAR} resumes. Such missing labels count correct predictions as false positives and teach the model to stay silent, which lowers both precision and recall.`));
add(p(`Preprocessing choices mattered less than expected for text normalisation (no measurable effect) but a great deal for the handling of long documents (sliding windows) and for the label conversion itself: without overlap resolution and word-boundary expansion, ${al.span_misaligned_expanded_to_word_boundary} spans would have produced invalid or partial BIO sequences.`));
add(h2('8.5 How API or Application Enhances the NLP Task'));
add(p(`The API turns the model into a usable tool: any resume file is converted to structured JSON in about a second, so it can feed an ATS or a search index. The web demo (${F('demo_compare')}) makes the effect of fine-tuning tangible: the side-by-side view with gold labels and per-document F1 lets a non-expert see which entities each model finds, and the Results tab presents the measured before-versus-after numbers next to the live predictions. The regex rule layer shows how simple domain knowledge can complement the neural model at the application level.`));

// ===== 9
add(h1('9. Conclusion'));
add(h2('9.1 Summary of the Work Done'));
add(p(`A complete resume-parsing system based on NER was built: data preparation with logged span alignment, sliding-window BERT inference, fine-tuning with validation-based model selection, ablations, 5-fold cross-validation, error analysis, a FastAPI service and a React demo. Using the same checkpoint, test split and metric, fine-tuning dslim/bert-base-NER on ${S.split_sizes.train} resumes raised entity-level micro-F1 from ${pct(before.micro.f1)} % to ${pct(after.micro.f1)} % (${pct(afterR.micro.f1)} % with the regex layer) and macro-F1 from ${pct(before.macro.f1)} % to ${pct(after.macro.f1)} %; cross-validation gives ${pct(P.cv_summary.all_types.micro_f1_mean)} % vs ${pct(ftCV.test_micro_f1.mean)} ± ${pct(ftCV.test_micro_f1.std)} %.`));
add(h2('9.2 Key Learnings and Insights'));
add(...bullets([
  'A pretrained NER model does not transfer to a new domain when the label set and annotation guidelines differ; most of its "knowledge" is useful only after fine-tuning re-maps it.',
  `Fine-tuning is data-efficient: ${Math.round(M.train_time_sec / 60)} minutes of training on ${S.split_sizes.train} resumes raise micro-F1 from ${pct(before.micro.f1)} % to ${pct(after.micro.f1)} %, because the encoder's language knowledge is reused.`,
  `Evaluation details matter: strict entity matching, a fixed split, validation-only selection and cross-validation change the story considerably compared with token accuracy (the model labels ${pct(rowN('O', 'O'))} % of O words correctly, yet entity F1 is under 50 %).`,
  'Data quality is the ceiling: many remaining "errors" are annotation gaps, and long, loosely defined entities such as SKILLS are hard to score with exact matching.',
  'Not every popular trick helps: class weighting reduced F1 here, while a simple sliding window and a regex rule did help.',
]));
add(h2('9.3 Real-World Applicability'));
add(p('The fine-tuned model already extracts names, locations, Indeed links, employers and degrees well enough to pre-fill candidate profiles for human review, and the API runs on commodity hardware without a GPU. For production use, it would need more and cleaner training data, an OCR step for scanned resumes, monitoring of extraction quality, data-protection safeguards and a bias audit, since automated screening tools can disadvantage candidates if they are not evaluated carefully [16].'));
add(h2('9.4 Future Improvements or Extensions'));
add(...bullets([
  '**Layout-aware models** such as LayoutLM [15], which combine text with 2-D position, to exploit columns, headings and tables in PDF resumes.',
  '**Larger and cleaner data:** re-annotating the dataset consistently (real e-mail addresses, all employers, skill items instead of whole sections) and adding more resumes from other countries and industries.',
  '**LLM-based extraction:** prompting or fine-tuning a generative LLM to output the JSON record directly, compared with the BERT tagger on the same test set for accuracy, cost and latency.',
  '**Better decoding:** a CRF layer or constrained decoding for valid BIO sequences, and relaxed (partial-match) metrics for long entities such as SKILLS.',
  '**Efficiency:** distillation or quantisation of the model for faster CPU inference.',
  '**Fairness and bias:** measuring whether extraction quality differs across name origins, genders or institutions before any use in hiring decisions [16].',
]));

// ===== 10
add(h1('10. References'));
const refs = [
  'D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft. [Online]. Available: https://web.stanford.edu/~jurafsky/slp3/',
  'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, pp. 5998–6008.',
  'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.',
  'E. F. Tjong Kim Sang and F. De Meulder, "Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition," in *Proc. CoNLL-2003*, 2003, pp. 142–147.',
  'dslim, "bert-base-NER," Hugging Face model card. [Online]. Available: https://huggingface.co/dslim/bert-base-NER (accessed Oct. 16, 2026).',
  'T. Wolf et al., "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP 2020: System Demonstrations*, 2020, pp. 38–45.',
  'Hugging Face, "Token classification," Transformers documentation. [Online]. Available: https://huggingface.co/docs/transformers/tasks/token_classification (accessed Oct. 16, 2026).',
  'H. Nakayama, "seqeval: A Python framework for sequence labeling evaluation," 2018. [Online]. Available: https://github.com/chakki-works/seqeval',
  'DataTurks, "Resume Entities for NER," Kaggle dataset. [Online]. Available: https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner (accessed Oct. 16, 2026).',
  'L. A. Ramshaw and M. P. Marcus, "Text chunking using transformation-based learning," in *Proc. 3rd Workshop on Very Large Corpora*, 1995, pp. 82–94.',
  'Y. Wu et al., "Google\'s neural machine translation system: Bridging the gap between human and machine translation," arXiv:1609.08144, 2016.',
  'G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer, "Neural architectures for named entity recognition," in *Proc. NAACL-HLT*, 2016, pp. 260–270.',
  'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.',
  'M. Mosbach, M. Andriushchenko, and D. Klakow, "On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines," in *Proc. ICLR*, 2021.',
  'Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, and M. Zhou, "LayoutLM: Pre-training of text and layout for document image understanding," in *Proc. ACM SIGKDD*, 2020, pp. 1192–1200.',
  'M. Raghavan, S. Barocas, J. Kleinberg, and K. Levy, "Mitigating bias in algorithmic hiring: Evaluating claims and practices," in *Proc. ACM Conf. Fairness, Accountability, and Transparency (FAT∗)*, 2020, pp. 469–481.',
  'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 8024–8035.',
  'Affinda, "Resume parser: Integration," Affinda documentation. [Online]. Available: https://docs.affinda.com/resumes/integration (accessed Oct. 16, 2026).',
  'Textkernel, "Tx Platform v9: Resume Parser – Getting started," developer documentation. [Online]. Available: https://developer.textkernel.com/tx-platform/v9/resume-parser/overview/getting-started/ (accessed Oct. 16, 2026).',
  'D. Hendrycks and K. Gimpel, "Gaussian error linear units (GELUs)," arXiv:1606.08415, 2016.',
  'J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," arXiv:1607.06450, 2016.',
];
refs.forEach((r, i) => add(new Paragraph({ spacing: { after: 100 }, indent: { left: 560, hanging: 560 },
  tabStops: [{ type: TabStopType.LEFT, position: 560 }], children: [new TextRun(`[${i + 1}]\t`), ...runs(r)] })));
add(new Paragraph({ spacing: { before: 240 }, children: runs('*Code, data-processing logs, metrics (the JSON files in results/) and figures are in the project repository; every number in this report is read from those files by report/build_report.js.*', { size: 20 }) }));

// ---------------------------------------------------------------- document
const footer = new Footer({ children: [new Paragraph({
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF', space: 4 } },
  children: [new TextRun({ text: `LLM AAT Report [${team.course_code}]`, font: 'Calibri', size: 20 }),
    new TextRun({ text: '\tPage ', font: 'Calibri', size: 20 }),
    new TextRun({ children: [PageNumber.CURRENT], bold: true, font: 'Calibri', size: 20 }),
    new TextRun({ text: ' of ', font: 'Calibri', size: 20 }),
    new TextRun({ children: [PageNumber.TOTAL_PAGES], bold: true, font: 'Calibri', size: 20 })],
})] });

const doc = new Document({
  creator: `${m1.name}, ${m2.name}`, title: team.topic, description: 'NLP for LLM AAT report',
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: '000000' }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: '000000' }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 340 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1300, bottom: 1300, left: MARGIN, right: MARGIN, footer: 600 } } },
    footers: { default: footer },
    children: [...cover, ...toc, ...secs],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, 'report.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out, `(${figs.length} figures, ${tabs.length} tables, ${refs.length} references)`);
});
