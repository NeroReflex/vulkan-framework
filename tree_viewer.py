#!/usr/bin/env python
import re
import sys
import json
from pathlib import Path
from collections import defaultdict, deque

def parse_tree(text: str):
    node_re = re.compile(
        r"node index (\d+) -- parent: (\d+), left: (\d+), right: (\d+)"
    )
    nodes = {}
    for line in text.splitlines():
        m = node_re.match(line.strip())
        if not m:
            continue
        idx, parent, left, right = map(int, m.groups())
        nodes[idx] = {"parent": parent, "left": left, "right": right}
    return nodes

def validate(nodes):
    errors = []
    warnings = []

    children = defaultdict(list)
    parents = defaultdict(list)

    for idx, data in nodes.items():
        p = data["parent"]
        if p in nodes:
            parents[idx].append(p)
            children[p].append(idx)
        # left
        l = data["left"]
        if l & 0x80000000 == 0 and l in nodes:  # internal child
            parents[l].append(idx)
            children[idx].append(l)
        # right
        r = data["right"]
        if r & 0x80000000 == 0 and r in nodes:
            parents[r].append(idx)
            children[idx].append(r)

    # multiple parents
    for n, ps in parents.items():
        if len(set(ps)) > 1:
            errors.append(f"Node {n} has multiple parents {ps}")

    # find roots
    roots = [n for n, d in nodes.items() if d["parent"] == n]
    if not roots:
        errors.append("No root found")
    else:
        if len(roots) > 1:
            warnings.append(f"Multiple roots? {roots}")

    # reachability
    reachable = set()
    if roots:
        q = deque([roots[0]])
        while q:
            x = q.popleft()
            if x in reachable:
                continue
            reachable.add(x)
            q.extend(children[x])
    unreachable = set(nodes) - reachable
    if unreachable:
        errors.append(f"Unreachable nodes: {sorted(unreachable)[:20]}{'...' if len(unreachable)>20 else ''}")

    return errors, warnings

def generate_html(nodes, output_path="tree.html"):
    # Convert to format usable by d3.js
    def make_node(idx):
        if idx not in nodes:
            return {"name": str(idx), "leaf": True}
        data = nodes[idx]
        is_leaf_left = bool(data["left"] & 0x80000000)
        is_leaf_right = bool(data["right"] & 0x80000000)
        children = []
        if data["left"]:
            if is_leaf_left:
                children.append({"name": str(data["left"] & 0x7FFFFFFF), "leaf": True})
            else:
                children.append(make_node(data["left"]))
        if data["right"]:
            if is_leaf_right:
                children.append({"name": str(data["right"] & 0x7FFFFFFF), "leaf": True})
            else:
                children.append(make_node(data["right"]))
        return {"name": str(idx), "children": children, "leaf": False}

    roots = [n for n, d in nodes.items() if d["parent"] == n]
    if not roots:
        root = list(nodes.keys())[0]
    else:
        root = roots[0]

    tree_data = make_node(root)

    html = f"""
<!DOCTYPE html>
<meta charset="utf-8">
<style>
.node circle {{ fill: #999; }}
.node.leaf circle {{ fill: red; }}
.link {{ fill: none; stroke: #555; stroke-opacity: 0.4; stroke-width: 1.5px; }}
</style>
<body>
<div id="tree"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {json.dumps(tree_data)};

const width = 1600, dx = 15, dy = 40;
const tree = d3.tree().nodeSize([dx, dy]);
const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);

const root = d3.hierarchy(data);
tree(root);

let x0 = Infinity, x1 = -x0;
root.each(d => {{ if (d.x > x1) x1 = d.x; if (d.x < x0) x0 = d.x; }});

const svg = d3.create("svg")
    .attr("viewBox", [0, 0, width, x1 - x0 + dx * 2])
    .style("font", "10px sans-serif")
    .style("user-select", "none");

const g = svg.append("g").attr("transform", `translate(40,${{dx - x0}})`);

g.append("g")
  .attr("fill", "none")
  .attr("stroke", "#555")
  .attr("stroke-opacity", 0.4)
  .attr("stroke-width", 1.5)
.selectAll("path")
.data(root.links())
.join("path")
  .attr("d", diagonal);

const node = g.append("g")
  .attr("stroke-linejoin", "round")
  .attr("stroke-width", 3)
.selectAll("g")
.data(root.descendants())
.join("g")
   .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

node.append("circle")
  .attr("r", 4)
  .attr("fill", d => d.data.leaf ? "red" : "#999");

node.append("text")
  .attr("dy", "0.31em")
  .attr("x", d => d.children ? -6 : 6)
  .attr("text-anchor", d => d.children ? "end" : "start")
  .text(d => d.data.name)
  .clone(true).lower()
  .attr("stroke", "white");

document.body.appendChild(svg.node());
</script>
"""
    Path(output_path).write_text(html)
    print(f"HTML visualization written to {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python tree_viewer.py tree.txt")
        return
    text = Path(sys.argv[1]).read_text()
    nodes = parse_tree(text)
    errors, warnings = validate(nodes)
    print("Validation Summary:")
    if errors:
        print("Errors:")
        for e in errors:
            print("  -", e)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print("  -", w)
    if not errors and not warnings:
        print("Tree looks valid ✅")
    generate_html(nodes, "tree.html")

if __name__ == "__main__":
    main()
