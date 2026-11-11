# Build settings for this report, read by every `latexmk` run in this folder -- including
# LaTeX Workshop's "latexmk (latexmkrc)" recipe, which passes no options of its own.
$pdf_mode = 1;                       # pdflatex: the figures are PNGs, which DVI-mode `latex` cannot read
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';   # never stop at a "?" prompt
