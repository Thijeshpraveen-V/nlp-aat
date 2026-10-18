import { useEffect, useMemo, useState } from 'react'

const TYPES = [
  'NAME', 'EMAIL', 'LOCATION', 'COLLEGE_NAME', 'DEGREE', 'GRADUATION_YEAR',
  'COMPANIES_WORKED_AT', 'DESIGNATION', 'YEARS_OF_EXPERIENCE', 'SKILLS',
]
const PRETTY = {
  NAME: 'Name', EMAIL: 'Email', LOCATION: 'Location', COLLEGE_NAME: 'College',
  DEGREE: 'Degree', GRADUATION_YEAR: 'Grad. year', COMPANIES_WORKED_AT: 'Company',
  DESIGNATION: 'Designation', YEARS_OF_EXPERIENCE: 'Years exp.', SKILLS: 'Skills',
}
const MODELS = [
  { key: 'pretrained', label: 'Pretrained (before)' },
  { key: 'finetuned', label: 'Fine-tuned (after)' },
]

async function api(path, opts) {
  const r = await fetch(path, opts)
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail || msg } catch { /* not json */ }
    throw new Error(msg)
  }
  return r.json()
}

function Highlighted({ text, entities }) {
  const parts = []
  let pos = 0
  ;[...entities].sort((a, b) => a.start - b.start).forEach((e, i) => {
    if (e.start < pos) return
    if (e.start > pos) parts.push(text.slice(pos, e.start))
    parts.push(
      <mark key={i} className={`ent ent-${e.label}`} title={e.label}>
        {text.slice(e.start, e.end)}
        <span className="tag">{PRETTY[e.label]}</span>
      </mark>,
    )
    pos = e.end
  })
  parts.push(text.slice(pos))
  return <div className="doc">{parts}</div>
}

function docScore(pred, gold) {
  if (!gold) return null
  const key = (e) => `${e.start}:${e.end}:${e.label}`
  const g = new Set(gold.map(key))
  const tp = pred.filter((e) => g.has(key(e))).length
  const p = pred.length ? tp / pred.length : 0
  const r = gold.length ? tp / gold.length : 0
  return { p, r, f1: p + r ? (2 * p * r) / (p + r) : 0, tp }
}

function Structured({ res }) {
  const rows = [
    ['Name', res.name], ['Email', res.email], ['Phone', res.phone],
    ['Location', res.location?.join(', ')],
    ['Companies', res.companies?.join('; ')], ['Designations', res.designations?.join('; ')],
    ['Degrees', res.education?.degrees.join('; ')], ['Colleges', res.education?.colleges.join('; ')],
    ['Grad. years', res.education?.graduation_years.join(', ')],
    ['Experience', res.years_of_experience?.join(', ')],
    ['Skills', res.skills?.join(', ')],
  ]
  return (
    <table className="fields">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}><th>{k}</th><td>{v || <span className="muted">—</span>}</td></tr>
        ))}
      </tbody>
    </table>
  )
}

function ModelPanel({ title, res, gold, view }) {
  if (!res) return null
  const s = docScore(res.entities, gold)
  return (
    <section className="panel">
      <header className="panel-head">
        <h3>{title}</h3>
        <div className="chips">
          <span className="chip">{res.entities.length} entities</span>
          {res.latency_ms != null && <span className="chip">{res.latency_ms.toFixed(0)} ms CPU</span>}
          {s && <span className="chip strong">doc F1 {(s.f1 * 100).toFixed(1)}%</span>}
        </div>
      </header>
      {view === 'text' ? <Highlighted text={res.text} entities={res.entities} /> : <Structured res={res} />}
    </section>
  )
}

function Legend() {
  return (
    <div className="legend">
      {TYPES.map((t) => <span key={t} className={`ent ent-${t}`}>{PRETTY[t]}</span>)}
    </div>
  )
}

function Demo() {
  const [samples, setSamples] = useState([])
  const [sample, setSample] = useState('')
  const [text, setText] = useState('')
  const [gold, setGold] = useState(null)
  const [file, setFile] = useState(null)
  const [rules, setRules] = useState(true)
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const [mode, setMode] = useState(params.get('mode') || 'compare')
  const [view, setView] = useState(params.get('view') || 'text')
  const [results, setResults] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => {
    api('/samples').then(setSamples).catch((e) => setErr(e.message))
    // deep link: ?sample=<name> loads and parses a sample (used for screenshots / live demo)
    const s = params.get('sample')
    if (s) pickSample(s).then((t) => run(t)).catch((e) => setErr(e.message))
  }, [params]) // eslint-disable-line react-hooks/exhaustive-deps

  async function pickSample(name) {
    setSample(name); setFile(null); setResults(null); setErr('')
    if (!name) { setText(''); setGold(null); return '' }
    const s = await api(`/samples/${encodeURIComponent(name)}`)
    setText(s.text); setGold(s.gold)
    return s.text
  }

  async function run(textOverride) {
    const text_ = typeof textOverride === 'string' ? textOverride : text
    setBusy(true); setErr('')
    try {
      let r
      if (file) {
        const fd = new FormData()
        fd.append('file', file); fd.append('model', 'both'); fd.append('rules', rules)
        r = await api('/parse', { method: 'POST', body: fd })
        setText(r.results.finetuned.text)
      } else {
        r = await api('/parse_text', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text_, model: 'both', rules }),
        })
      }
      setResults(r.results)
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  const shown = mode === 'compare' ? MODELS : MODELS.filter((m) => m.key === mode)
  const goldRes = gold && results ? { ...results.finetuned, entities: gold, latency_ms: null } : null

  return (
    <>
      <section className="card controls">
        <div className="row">
          <label className="field">
            <span>Sample resume</span>
            <select value={sample} onChange={(e) => pickSample(e.target.value)}>
              <option value="">— choose —</option>
              {samples.map((s) => (
                <option key={s.name} value={s.name}>{s.name}{s.has_gold ? ' (gold labels)' : ''}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>…or upload PDF / DOCX / TXT</span>
            <input type="file" accept=".pdf,.docx,.txt"
              onChange={(e) => { setFile(e.target.files[0] || null); setSample(''); setGold(null); setResults(null) }} />
          </label>
        </div>
        {!file && (
          <textarea value={text} onChange={(e) => { setText(e.target.value); setGold(null); setSample('') }}
            placeholder="…or paste resume text here" rows={6} />
        )}
        <div className="row actions">
          <div className="seg" role="group" aria-label="Model">
            {[['pretrained', 'Pretrained'], ['finetuned', 'Fine-tuned'], ['compare', 'Side by side']].map(([k, l]) => (
              <button key={k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}>{l}</button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="View">
            {[['text', 'Highlighted text'], ['fields', 'Structured fields']].map(([k, l]) => (
              <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>{l}</button>
            ))}
          </div>
          <label className="check">
            <input type="checkbox" checked={rules} onChange={(e) => setRules(e.target.checked)} />
            Regex rule layer (fine-tuned)
          </label>
          <button className="primary" disabled={busy || (!file && !text.trim())} onClick={run}>
            {busy ? 'Parsing…' : 'Parse resume'}
          </button>
        </div>
        {err && <p className="error">{err}</p>}
      </section>

      {results && (
        <>
          <Legend />
          <div className={`grid cols-${shown.length + (goldRes ? 1 : 0)}`}>
            {goldRes && <ModelPanel title="Gold annotation" res={goldRes} view={view} />}
            {shown.map((m) => (
              <ModelPanel key={m.key} title={m.label} res={results[m.key]} gold={gold} view={view} />
            ))}
          </div>
        </>
      )}
    </>
  )
}

const pct = (v) => (v == null ? '—' : (v * 100).toFixed(1))

function Results() {
  const [m, setM] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => { api('/metrics').then(setM).catch((e) => setErr(e.message)) }, [])
  const max = useMemo(() => 1, [])
  if (err) return <p className="error">{err}</p>
  if (!m) return <p className="muted">Loading…</p>
  const ba = m.before_after
  return (
    <>
      <section className="card">
        <h2>Before vs after — held-out test split ({m.n_test_resumes} resumes)</h2>
        <p className="muted">Entity-level F1 (seqeval, strict IOB2). Before = {m.base_model} used as-is with
          PER→NAME, ORG→COMPANIES_WORKED_AT, LOC→LOCATION. After = same checkpoint fine-tuned ({m.final_run}).</p>
        <div className="kpis">
          {ba.overall.map((o) => (
            <div key={o.metric} className="kpi">
              <div className="kpi-label">{o.metric}</div>
              <div className="kpi-val">{pct(o.after)}<small>%</small></div>
              <div className="kpi-sub">before {pct(o.before)}% · Δ +{pct(o.delta)} pts</div>
            </div>
          ))}
        </div>
        <table className="metrics">
          <thead><tr><th>Entity type</th><th>Support</th><th>Before F1</th><th>After F1</th><th>Δ</th><th className="barcol">F1 (before / after)</th></tr></thead>
          <tbody>
            {ba.rows.map((r) => (
              <tr key={r.type}>
                <td>{PRETTY[r.type] || r.type}</td><td>{r.support}</td>
                <td>{pct(r.before)}</td><td><b>{pct(r.after)}</b></td><td>+{pct(r.delta)}</td>
                <td className="barcol">
                  <div className="bar before" style={{ width: `${(r.before / max) * 100}%` }} />
                  <div className="bar after" style={{ width: `${(r.after / max) * 100}%` }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      {m.cv && (
        <section className="card">
          <h2>5-fold cross-validation (mean ± std)</h2>
          <table className="metrics">
            <thead><tr><th>Model</th><th>Micro F1</th><th>Macro F1</th></tr></thead>
            <tbody>
              {m.cv.map((r) => (
                <tr key={r.model}><td>{r.model}</td>
                  <td>{pct(r.micro_mean)} ± {pct(r.micro_std)}</td><td>{pct(r.macro_mean)} ± {pct(r.macro_std)}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {m.ablations && (
        <section className="card">
          <h2>Ablations (test split)</h2>
          <table className="metrics">
            <thead><tr><th>Setting</th><th>Micro F1</th><th>Macro F1</th></tr></thead>
            <tbody>
              {m.ablations.map((r) => (
                <tr key={r.name}><td>{r.name}</td><td>{pct(r.micro_f1)}</td><td>{pct(r.macro_f1)}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {m.figures?.length > 0 && (
        <section className="card figs">
          {m.figures.map((f) => <figure key={f}><img src={`/figures/${f}`} alt={f} /><figcaption>{f}</figcaption></figure>)}
        </section>
      )}
    </>
  )
}

export default function App() {
  const [tab, setTabState] = useState(window.location.hash === '#results' ? 'results' : 'demo')
  const setTab = (t) => { setTabState(t); window.history.replaceState(null, '', t === 'results' ? '#results' : '#') }
  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Resume NER</h1>
          <p className="muted">Pretrained vs fine-tuned BERT (dslim/bert-base-NER) for resume parsing</p>
        </div>
        <nav className="seg">
          <button className={tab === 'demo' ? 'on' : ''} onClick={() => setTab('demo')}>Live demo</button>
          <button className={tab === 'results' ? 'on' : ''} onClick={() => setTab('results')}>Results</button>
        </nav>
      </header>
      <main>{tab === 'demo' ? <Demo /> : <Results />}</main>
    </div>
  )
}
