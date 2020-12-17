#set terminal pdfcairo 
set encoding utf8

#set output "temp2.pdf"

set title "{/:Bold Temperature Measurement}" font "sans-Bold"
set xlabel "{/:Bold Time [day]}"
set ylabel "{/:Bold Temperature [°C]}"

#set datafile separator ","

set xrange [1:32]
set yrange [18:22]

#remove top & right axes
#set xtics nomirror
#set ytics nomirror

set style line 1 lt 1 lw 3 lc rgb "#FFAA00" pt 7 ps 2
set style line 2 lt 1 lw 2 lc rgb "#7000AEFF"

f(x) = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9 + k*x**10 + l*x**11 + m*x**12 + n*x**13
fit f(x) "temp3.txt" index 0 using 1:2 via a,b,c,d,e,f,g,h,i,j,k,l,m,n

show style line
plot "temp3.txt" title "Teplota" with linespoints ls 1, f(x) title "F(x)" with lines ls 2