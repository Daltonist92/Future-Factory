const canvas = document.getElementById('factoryCanvas');
const ctx = canvas.getContext('2d');
const GRID_SIZE = 30;
const GRID_WIDTH = canvas.width / GRID_SIZE;
const GRID_HEIGHT = canvas.height / GRID_SIZE;
const machines = [{ id: 'WS1', x: 7, y: 1 }];
const inventories = [{ id: 'INV1', x: 2, y: 2 }];
const arrows = [];
let turns = 0;
let dragging = null;
let flowMode = false;
let flowStart = null;
let lastDeleted = null;

function getRandomPosition(exclude) {
    let x, y;
    do {
        x = Math.floor(Math.random() * (GRID_WIDTH - 2)) + 1;
        y = Math.floor(Math.random() * (GRID_HEIGHT - 2)) + 1;
    } while (exclude.some(item => item.x === x && item.y === y));
    return { x, y };
}

function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#ddd';
    for (let x = 0; x <= canvas.width; x += GRID_SIZE) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += GRID_SIZE) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

function getDotPositions(item) {
    const x = item.x * GRID_SIZE + GRID_SIZE / 2;
    const y = item.y * GRID_SIZE + GRID_SIZE / 2;
    const isWS = item.id.startsWith('WS');
    if (isWS) {
        return {
            top: { x, y: y - 40 },
            bottom: { x, y: y + 40 },
            left: { x: x - 40, y },
            right: { x: x + 40, y }
        };
    } else {
        return {
            top: { x, y: y - 30 },
            bottom: { x, y: y + 30 },
            left: { x: x - 25, y },
            right: { x: x + 25, y }
        };
    }
}

function drawLayout() {
    drawGrid();
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';

    machines.forEach(m => {
        const x = m.x * GRID_SIZE + GRID_SIZE / 2;
        const y = m.y * GRID_SIZE + GRID_SIZE / 2;
        ctx.strokeStyle = dragging && dragging.id === m.id ? 'red' : 'black';
        ctx.strokeRect(x - 40, y - 40, 80, 80);
        ctx.fillText(m.id, x, y);
        const dots = getDotPositions(m);
        for (let pos in dots) {
            ctx.beginPath();
            ctx.arc(dots[pos].x, dots[pos].y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = 'black';
            ctx.fill();
        }
    });

    inventories.forEach(i => {
        const x = i.x * GRID_SIZE + GRID_SIZE / 2;
        const y = i.y * GRID_SIZE + GRID_SIZE / 2;
        ctx.beginPath();
        ctx.moveTo(x, y - 30);
        ctx.lineTo(x + 50, y + 30);
        ctx.lineTo(x - 50, y + 30);
        ctx.closePath();
        ctx.strokeStyle = dragging && dragging.id === i.id ? 'red' : 'black';
        ctx.stroke();
        ctx.fillText(i.id, x, y - 5);
        const dots = getDotPositions(i);
        for (let pos in dots) {
            ctx.beginPath();
            ctx.arc(dots[pos].x, dots[pos].y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = 'black';
            ctx.fill();
        }
    });

    arrows.forEach(arrow => {
        const startItem = [...machines, ...inventories].find(item => item.id === arrow.startId);
        const endItem = [...machines, ...inventories].find(item => item.id === arrow.endId);
        if (startItem && endItem) {
            const startDots = getDotPositions(startItem);
            const endDots = getDotPositions(endItem);
            const start = startDots[arrow.startPos];
            const end = endDots[arrow.endPos];
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();
            const angle = Math.atan2(end.y - start.y, end.x - start.x);
            ctx.beginPath();
            ctx.moveTo(end.x, end.y);
            ctx.lineTo(end.x - 10 * Math.cos(angle - Math.PI / 6), end.y - 10 * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(end.x - 10 * Math.cos(angle + Math.PI / 6), end.y - 10 * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fillStyle = 'black';
            ctx.fill();
        }
    });

    if (flowMode && flowStart) {
        const mouseX = canvas._mouseX || flowStart.x;
        const mouseY = canvas._mouseY || flowStart.y;
        ctx.beginPath();
        ctx.moveTo(flowStart.x, flowStart.y);
        ctx.lineTo(mouseX, mouseY);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function runTurn() {
    turns++;
    console.log(`Turn ${turns}`);
    drawLayout();
}

function toggleAuto() {
    console.log('Toggle Auto (TODO)');
}

function addWorkstation() {
    const pos = getRandomPosition(machines.concat(inventories));
    machines.push({ id: `WS${machines.length + 1}`, x: pos.x, y: pos.y });
    drawLayout();
}

function addInventory() {
    const pos = getRandomPosition(machines.concat(inventories));
    inventories.push({ id: `INV${inventories.length + 1}`, x: pos.x, y: pos.y });
    drawLayout();
}

function undoDelete() {
    if (!lastDeleted) {
        console.log('Nothing to undo');
        return;
    }

    const { type, item, deletedArrows } = lastDeleted;
    if (type === 'machine') {
        machines.push(item);
        console.log(`Restored ${item.id} at (${item.x}, ${item.y})`);
    } else if (type === 'inventory') {
        inventories.push(item);
        console.log(`Restored ${item.id} at (${item.x}, ${item.y})`);
    } else if (type === 'arrow') {
        arrows.push(item);
        console.log(`Restored arrow from ${item.startId} to ${item.endId}`);
    }

    if (deletedArrows) {
        deletedArrows.forEach(arrow => {
            arrows.push(arrow);
            console.log(`Restored arrow from ${arrow.startId} to ${arrow.endId}`);
        });
    }

    lastDeleted = null;
    drawLayout();
}

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (e.button === 0) {
        for (let item of [...machines, ...inventories]) {
            const dots = getDotPositions(item);
            for (let pos in dots) {
                const dx = dots[pos].x;
                const dy = dots[pos].y;
                if (Math.sqrt((x - dx) ** 2 + (y - dy) ** 2) < 6) {
                    if (flowMode && flowStart) {
                        if (flowStart.id !== item.id) {
                            arrows.push({
                                startId: flowStart.id,
                                endId: item.id,
                                startPos: flowStart.pos,
                                endPos: pos
                            });
                            console.log(`Arrow added: ${flowStart.id} (${flowStart.pos}) -> ${item.id} (${pos})`);
                        }
                        flowMode = false;
                        flowStart = null;
                        drawLayout();
                        return;
                    } else {
                        flowMode = true;
                        flowStart = { id: item.id, x: dx, y: dy, pos: pos };
                        canvas._mouseX = x;
                        canvas._mouseY = y;
                        drawLayout();
                        return;
                    }
                }
            }
        }

        for (let m of machines) {
            const mx = m.x * GRID_SIZE + GRID_SIZE / 2;
            const my = m.y * GRID_SIZE + GRID_SIZE / 2;
            if (Math.abs(mx - x) < 40 && Math.abs(my - y) < 40) {
                dragging = m;
                document.getElementById('info-id').textContent = m.id;
                document.getElementById('info-tool').textContent = 'Casting Shop';
                document.getElementById('info-state').textContent = 'Idle S:1';
                document.getElementById('info-oee').textContent = 'N/A';
                drawLayout();
                return;
            }
        }
        for (let i of inventories) {
            const ix = i.x * GRID_SIZE + GRID_SIZE / 2;
            const iy = i.y * GRID_SIZE + GRID_SIZE / 2;
            if (Math.abs(ix - x) < 50 && Math.abs(iy - y) < 30) {
                dragging = i;
                document.getElementById('info-id').textContent = i.id;
                document.getElementById('info-tool').textContent = 'Inventory Rack';
                document.getElementById('info-state').textContent = 'N/A';
                document.getElementById('info-oee').textContent = 'N/A';
                drawLayout();
                return;
            }
        }
    }
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    canvas._mouseX = x;
    canvas._mouseY = y;
    if (dragging) {
        dragging.x = Math.round(x / GRID_SIZE);
        dragging.y = Math.round(y / GRID_SIZE);
        dragging.x = Math.max(1, Math.min(dragging.x, GRID_WIDTH - 2));
        dragging.y = Math.max(1, Math.min(dragging.y, GRID_HEIGHT - 2));
        drawLayout();
    } else if (flowMode && flowStart) {
        drawLayout();
    }
});

canvas.addEventListener('mouseup', (e) => {
    if (dragging && e.button === 0) {
        console.log(`Moved ${dragging.id} to (${dragging.x}, ${dragging.y})`);
        dragging = null;
        drawLayout();
    }
});

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Cancel arrow drawing if in flow mode
    if (flowMode && flowStart) {
        console.log(`Arrow cancelled at ${flowStart.id} (${flowStart.pos})`);
        flowMode = false;
        flowStart = null;
        drawLayout();
        return;
    }

    // Delete logic
    for (let i = machines.length - 1; i >= 0; i--) {
        const m = machines[i];
        const mx = m.x * GRID_SIZE + GRID_SIZE / 2;
        const my = m.y * GRID_SIZE + GRID_SIZE / 2;
        if (Math.abs(mx - x) < 40 && Math.abs(my - y) < 40) {
            const deletedArrows = arrows.filter(a => a.startId === m.id || a.endId === m.id);
            lastDeleted = { type: 'machine', item: machines.splice(i, 1)[0], deletedArrows };
            console.log(`Deleted ${m.id}`);
            for (let j = arrows.length - 1; j >= 0; j--) {
                if (arrows[j].startId === m.id || arrows[j].endId === m.id) {
                    console.log(`Deleted arrow from ${arrows[j].startId} to ${arrows[j].endId}`);
                    arrows.splice(j, 1);
                }
            }
            drawLayout();
            return;
        }
    }

    for (let i = inventories.length - 1; i >= 0; i--) {
        const inv = inventories[i];
        const ix = inv.x * GRID_SIZE + GRID_SIZE / 2;
        const iy = inv.y * GRID_SIZE + GRID_SIZE / 2;
        if (Math.abs(ix - x) < 50 && Math.abs(iy - y) < 30) {
            const deletedArrows = arrows.filter(a => a.startId === inv.id || a.endId === inv.id);
            lastDeleted = { type: 'inventory', item: inventories.splice(i, 1)[0], deletedArrows };
            console.log(`Deleted ${inv.id}`);
            for (let j = arrows.length - 1; j >= 0; j--) {
                if (arrows[j].startId === inv.id || arrows[j].endId === inv.id) {
                    console.log(`Deleted arrow from ${arrows[j].startId} to ${arrows[j].endId}`);
                    arrows.splice(j, 1);
                }
            }
            drawLayout();
            return;
        }
    }

    for (let i = arrows.length - 1; i >= 0; i--) {
        const arrow = arrows[i];
        const startItem = [...machines, ...inventories].find(item => item.id === arrow.startId);
        const endItem = [...machines, ...inventories].find(item => item.id === arrow.endId);
        if (startItem && endItem) {
            const startDots = getDotPositions(startItem);
            const endDots = getDotPositions(endItem);
            const start = startDots[arrow.startPos];
            const end = endDots[arrow.endPos];
            const midX = (start.x + end.x) / 2;
            const midY = (start.y + end.y) / 2;
            if (Math.sqrt((x - midX) ** 2 + (y - midY) ** 2) < 10) {
                lastDeleted = { type: 'arrow', item: arrows.splice(i, 1)[0], deletedArrows: null };
                console.log(`Deleted arrow from ${arrow.startId} to ${arrow.endId}`);
                drawLayout();
                return;
            }
        }
    }

    console.log('Right-click: Nothing to delete');
});

drawLayout();