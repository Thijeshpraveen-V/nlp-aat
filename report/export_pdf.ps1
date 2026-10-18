# Opens report.docx in Microsoft Word, updates the table of contents, saves the DOCX and
# exports report.pdf. Page numbers (PAGE / NUMPAGES fields) refresh when Word lays out the file.
#   powershell -ExecutionPolicy Bypass -File report/export_pdf.ps1
# Notes: TOC is accessed by index (enumerating the COM collection can hang), and
# Fields.Update() is avoided because it can block on a hidden dialog.
$docx = Join-Path $PSScriptRoot "report.docx"
$pdf = Join-Path $PSScriptRoot "report.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    $doc.TablesOfContents(1).Update()
    $doc.Repaginate()
    $doc.TablesOfContents(1).Update()   # second pass after pagination
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)   # wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close(0)
} finally {
    $word.Quit()
}
"exported $pdf"
