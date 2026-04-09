$(document).ready(function() {
    var canvas = document.getElementById("canvas");
    var context = canvas.getContext("2d");

    // Pre-fill with white (AI needs a white background to see your black lines)
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    var currentStrokeSize = 10;
    context.lineWidth = currentStrokeSize;
    context.lineJoin = context.lineCap = 'round';
    context.strokeStyle = "black";

    var painting = false;

    // Stroke size slider control
    $("#strokeSize").on("input", function() {
        currentStrokeSize = $(this).val();
        context.lineWidth = currentStrokeSize;
        $("#sizeValue").text(currentStrokeSize);
    });

    // This function calculates the EXACT position of the mouse on the canvas
    function getMousePos(e) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX || e.touches[0].clientX) - rect.left,
            y: (e.clientY || e.touches[0].clientY) - rect.top
        };
    }

    function startPosition(e) {
        painting = true;
        draw(e);
    }

    function finishedPosition() {
        painting = false;
        context.beginPath(); // This prevents lines from jumping across the screen
    }

    function draw(e) {
        if (!painting) return;

        var pos = getMousePos(e);

        context.lineTo(pos.x, pos.y);
        context.stroke();
        context.beginPath();
        context.moveTo(pos.x, pos.y);
    }

    // Event Listeners
    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("touchstart", startPosition); // Mobile support!
    
    canvas.addEventListener("mouseup", finishedPosition);
    canvas.addEventListener("touchend", finishedPosition);
    
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("touchmove", draw);

    // Clear Button Logic
    $("#clearButton").on("click", function() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "white";
        context.fillRect(0, 0, canvas.width, canvas.height);
        $("#chartWrapper").hide();
        $("#errorMessage").hide();
    });
});