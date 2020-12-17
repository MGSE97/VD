set terminal pdfcairo 

set output "cv_01b.pdf"

plot "data.txt" index 0 title "Outside" with linespoints, "data.txt" index 1 title "Inside" with lines
