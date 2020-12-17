document.addEventListener('DOMContentLoaded', function() {
  // set the dimensions and margins of the graph
  el = document.getElementById('plot');
  var width = el.clientWidth,
      height = el.clientHeight;

  // append the svg object to the body of the page
  var svg = d3.select("#plot")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  // #region Create data
  var lvls = 10;
  var lvlNodes = 10;
  var rnd_link = !true // Randomly set link to previous main node
  var split = 0.5      // Chance to cut connections between main nodes

  var step = lvlNodes+1
  var nodes = [];
  var links = [];
  for (var i = 0; i < lvls; i++) {
    main = nodes.length;

    // Main nodes
    nodes.push({
      lvl: i,
      main: true
    });
    
    // Subnodes
    for (var j = 0; j < lvlNodes; j++) {
      nodes.push({
        lvl: i,
        node: j
      });

      links.push({
        source: main + (rnd_link ? Math.round(Math.random() * step) : 0),
        target: main + j + 1,
        lvl: i,
        node: j
      });
    }

    // Link between levels
    if(i > 0 && Math.random() >= split)
    {
      links.push({
        source: i*step,
        target: (i-1)*step,
        lvl: i,
        node: 0
      });
    }
  }
  
  var graph = {
    nodes: nodes,
    links: links
  };

  // #endregion

  // Draw data
  var force = d3.layout.force()
    .nodes(graph.nodes)
    .links(graph.links)
    .size([width, height])
    .charge(-200)
    .linkDistance(50)
    .on("tick", tick)
    .start();

  var link = svg.selectAll(".link")
    .data(graph.links)
    .enter()
    .append("line")
    .attr("class", "link")
    .attr("stroke", d => colorLink(d.lvl, d.node));

  var node = svg.selectAll(".node")
    .data(graph.nodes)
    .enter()
    .append("g")
      .call(force.drag);

  var circle = node.append("circle")
      .attr("class", "node")
      .attr("r", 12)
      .attr("fill", d => colorNode(d.lvl))
      .attr("stroke", d => d.main ? "#0F0" : "");

  var text = node.append("text")
      .attr("class", "text")
      .attr("dx", d => d.main ? "-0.25em" : "-0.65em")
      .attr("dy", ".35em")
      .text(d => `${d.lvl}${d.node >= 0 ? `.${d.node}` : ''}`);

  function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
      const hex = x.toString(16)
      return hex.length === 1 ? '0' + hex : hex
    }).join('');
  }

  function colorLink(lvl, node) {
    a = lvl * Math.floor(255/lvls);
    if(rnd_link)
      return rgbToHex(255, 165, a);
    
    b = node * Math.floor(255/lvlNodes);
    return rgbToHex(255, b, a);
  }
  function colorNode(lvl) {
    v = lvl * Math.floor(255/lvls);
    return rgbToHex(255, 165, v);
  }

  function tick() {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    circle
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);

    text
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  }
});