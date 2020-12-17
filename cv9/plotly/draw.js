const groupBy = (items, key) => items.reduce(
  (result, item) => ({
    ...result,
    [item[key]]: [
      ...(result[item[key]] || []),
      item,
    ],
  }), 
  {},
);

document.addEventListener('DOMContentLoaded', function() {
  fetch('commodities.json')
    .then(r => r.json())
    .then(data => {
      var d = data;
      fetch('commodities.fm')
      .then(r => r.json())
      .then(fm => {
        console.table(d.slice(0, 10));

        var data = d.filter(x => x.max_buy_price > 0 && x.max_sell_price > 0).map((x, i) => {
          x.fm = fm[i];
          return x;
        });

        var series = groupBy(data, 'category_id')
        var data = []
        for(var key in series)
        {
          var items = series[key];
          var s = {
            x: [],
            y: [],
            mode: 'markers',
            type: 'scatter',
            name: key,
            text: [],
            marker: { size: 10 }
          }
          for(var k in items)
          {
            let item = items[k];
            // choose x,y values
            let ix = item.fm.x;
            let iy = item.fm.y;
            s.x.push(ix);
            s.y.push(iy);
            s.text.push(item.name);
            s.name = item.category.name;
          }
          data.push(s);
        }
        console.table(data);
        
        var layout = {
          xaxis: {
            type: 'log',
            title: {
              text: 'Sell'
            }
          },
          yaxis: {
            type: 'log',
            title: {
              text: 'Buy'
            }
          },
          title:'Commodities',
        };
        
        Plotly.newPlot('plot', data, layout);
      });
    });
});