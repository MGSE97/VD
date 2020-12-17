#set terminal pdfcairo 
set encoding utf8

set output "temp.pdf"

set title "Temperature Measurement" font "sans-Bold"
set xlabel "Time (day)"
set ylabel "Temp (°C)"

set datafile separator ","

set xdata time
set timefmt "%Y%m%d %H%M"
set xrange ["20191231 1600":"20200101 2300"]
set yrange [18:22]

set format x "%d.%m.%y\n%H:%M"

#remove top & right axes
#set xtics nomirror
#set ytics nomirror

set style line 1 lt 1 lc rgb "#FF7607"

#f(x) = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9 + k*x**10 + l*x**11 + m*x*12 + n*x*13
#fit f(x) "temp2.txt" index 0 using 1:2 via a,b,c,d,e,f,g,h,i,j,k,l,m,n

#plot "temp2.txt" index 0 using (timecolumn(1, "%Y%m%d %H%M")):2 title "Teplota" with linespoints lc rgb "#FF7607", f(x) using (timecolumn(1, "%Y%m%d %H%M")):2 title "F(x)"

plot "temp2.txt" index 0 using (timecolumn(1, "%Y%m%d %H%M")):2 title "Teplota" with linespoints lc rgb "#FF7607"

#show output

#pause -1