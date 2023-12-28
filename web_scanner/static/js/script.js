document.addEventListener("click", function (e) {
    if (e.target.classList.contains("result-item")) {
        var parent = e.target.parentElement;
        
        const str = parent.querySelector('.card-img-top').getAttribute("alt");
        const data = JSON.parse(str.replaceAll("\'", "\"", ));

        document.querySelector("#viewer-title").innerText = data['image'].split('/')[data['image'].split('/').length - 1]; 
        document.querySelector("#viewer-barcode").innerHTML = parent.getAttribute("alt");;
        document.querySelector("#viewer-canvas").alt = str;
        document.querySelector("#viewer-date-field").innerHTML = data['time'];
        document.querySelector("#viewer-scanner-field").innerHTML = data['scanner'];
        document.querySelector("#viewer-user-field").innerHTML = data['user'];
    

        var myCanvas = document.getElementById('viewer-canvas');
        var context = myCanvas.getContext('2d');
        var img = new Image;
        img.onload = function(){
            var canvas = context.canvas ;
            var hRatio = canvas.width  / img.width;
            var vRatio =  canvas.height / img.height;
            var ratio  = Math.min ( hRatio, vRatio );
            var centerShift_x = ( canvas.width - img.width*ratio ) / 2;
            var centerShift_y = ( canvas.height - img.height*ratio ) / 2;  
            context.clearRect(0,0,canvas.width, canvas.height);
            context.drawImage(img, 0,0, img.width, img.height,
            centerShift_x,centerShift_y,img.width*ratio, img.height*ratio);  
        };
        img.src = data['image'];

        const myModal = new bootstrap.Modal(document.getElementById('viewer-modal'));
        myModal.show();
    }
});

document.addEventListener("click", function (e) {
    if (e.target.id == "viewer-show") {
        const data = JSON.parse(document.querySelector("#viewer-canvas").alt.replaceAll("\'", "\"", ));
        var canvas = document.getElementById('viewer-canvas');
        var context = canvas.getContext('2d');
        if (e.target.classList.contains("active")) {
            var img = new Image;
            img.onload = function(){
                var canvas = context.canvas ;
                var hRatio = canvas.width  / img.width;
                var vRatio =  canvas.height / img.height;
                var ratio  = Math.min ( hRatio, vRatio );
                var centerShift_x = ( canvas.width - img.width*ratio ) / 2;
                var centerShift_y = ( canvas.height - img.height*ratio ) / 2;  
                context.clearRect(0,0,canvas.width, canvas.height);
                context.drawImage(img, 0,0, img.width, img.height,
                centerShift_x,centerShift_y,img.width*ratio, img.height*ratio);  
            };
            img.src = data['image'];
            e.target.classList.remove('active');
        } else {
            context.beginPath();
            var img = new Image;
            img.src = data['image'];
            for (var i = 0; i < data['bboxes'].length; i++) {
                NewTop    = ((data['bboxes'][i][0]) * canvas.height / img.height);
                NewLeft   = ((data['bboxes'][i][1]) * canvas.width  / img.width);

                NewBottom = ((data['bboxes'][i][0] + data['bboxes'][i][2] + 1) * canvas.height / img.height) - 1;
                NewRight  = ((data['bboxes'][i][1] + data['bboxes'][i][3] + 1) * canvas.width  / img.width) - 1;
                //context.rect(data['bboxes'][i][0], data['bboxes'][i][1], data['bboxes'][i][2], data['bboxes'][i][3]);
                context.rect(NewTop, NewLeft, NewBottom - NewTop, NewRight - NewLeft);
                context.lineWidth = 5;
                context.strokeStyle = 'green';
                context.stroke();
            }
            e.target.classList.add('active');
        }
    }
});

document.addEventListener("click", function (e) {
    var elem = (e.target.id.startsWith("user-") && e.target.id.endsWith("-tab")) ? e.target : e.target.parentElement;
    if (elem.id.startsWith("user-") && elem.id.endsWith("-tab")) {
        var username = elem.id.split("-")[1];
        var tabs = document.getElementsByClassName('tab-pane');
        for (var i = 0; i < tabs.length; i++) {
            if ((tabs[i].id.startsWith("user-")) && !(tabs[i].id.includes(username))) {
                tabs[i].classList.remove("active");
            } else if (tabs[i].id.includes(username)) {
                tabs[i].classList.add("active");
            }
        }
    }
});

document.addEventListener("click", function (e) {
    if ((e.target.id.includes("save-user-")) || (e.target.id.includes("cancel-user-"))) {
        uid = e.target.id.split('-')[2];
        var username = document.getElementById(`user-${uid}-login`);
        var password = document.getElementById(`user-${uid}-password`);
        var name = document.getElementById(`user-${uid}-name`);
        var info = document.getElementById(`user-${uid}-job`);
        var role = document.getElementById(`user-${uid}-role`);
        var department = document.getElementById(`user-${uid}-department`);
        var scanners = document.getElementById(`user-${uid}-scanners`);
        var selected_scanners = [];
        for (const child of scanners.children) {
            if (child.children[0].checked) {
                selected_scanners.push(child.children[0].id.split("-")[2]);
            }
        }
        var user = {
            "id": uid, 
            "username": username.value, 
            "scanners": selected_scanners.join(', '),
            "name": name.value, 
            "role": role.selectedIndex + 1, 
            "info": info.value,
            "department": department.value, 
            "create": uid == "newuser"
        };
        if (uid == "newuser") {
            user['password'] = password.value;
        }
        if (e.target.id.includes("save-user-")) {
            $.getJSON('/update_user', user, traditional=true, function(data) {
                document.getElementById("main-toast-text").innerText = "Изменения внесены успешно";
                document.getElementById("main-toast").classList.remove("hide");
                document.getElementById("main-toast").classList.add("show");
                setInterval('location.reload()', 1000);
            });
        } else {
            // TODO: cancel
            setInterval('location.reload()', 1000);
        }
    }
});


function draw() {
    var Radius = 20;
    var squaresPerCircle = 20;

    const canvases = document.getElementsByClassName("scanner-canvas");
    for(var i = 0; i < canvases.length; i++) {
        var canvas = canvases[i];

        var ctx = canvas.getContext("2d");
        canvas.width = 65;
        canvas.height = 65;
        
        ctx.save();
        ctx.translate(canvas.width / 2, canvas.height / 2);

        var angle = 2 * Math.PI / squaresPerCircle;
        var squareSize = 2 * Math.PI * Radius / squaresPerCircle;
        ctx.save();
        for (var cIndex = 0; cIndex < squaresPerCircle; cIndex++) {
            ctx.fillStyle = "black";
            ctx.fillRect(Radius, -squareSize / 2, squareSize * 1.5, squareSize);
            ctx.rotate(angle);
        };

        ctx.restore();
    }
}


document.addEventListener("click", function (e) {
    if (e.target.id.startsWith("start-scanning")) {
        /*var parent = e.target.parentElement;
        
        const str = parent.querySelector('.card-img-top').getAttribute("alt");
        const data = JSON.parse(str.replaceAll("\'", "\"", ));

        document.querySelector("#viewer-title").innerText = data['image'].split('/')[data['image'].split('/').length - 1]; 
        document.querySelector("#viewer-barcode").innerHTML = parent.getAttribute("alt");;
        document.querySelector("#viewer-canvas").alt = str;
        document.querySelector("#viewer-date-field").innerHTML = data['time'];
        document.querySelector("#viewer-scanner-field").innerHTML = data['scanner'];
        document.querySelector("#viewer-user-field").innerHTML = data['user'];
    

        var myCanvas = document.getElementById('viewer-canvas');
        var context = myCanvas.getContext('2d');
        var img = new Image;
        img.onload = function(){
            var canvas = context.canvas ;
            var hRatio = canvas.width  / img.width;
            var vRatio =  canvas.height / img.height;
            var ratio  = Math.min ( hRatio, vRatio );
            var centerShift_x = ( canvas.width - img.width*ratio ) / 2;
            var centerShift_y = ( canvas.height - img.height*ratio ) / 2;  
            context.clearRect(0,0,canvas.width, canvas.height);
            context.drawImage(img, 0,0, img.width, img.height,
            centerShift_x,centerShift_y,img.width*ratio, img.height*ratio);  
        };
        img.src = data['image'];
        */
       /*
       if (e.target.id.includes("start-scanning-")) {
            sid = e.target.id.split('-')[2];
            $.getJSON('/start_scanning', {"scanner_id": sid}, traditional=true, function(data) {
                document.getElementById(`scanning-status-${sid}`).style = "color: red";
            });
        }
        */
        const myModal = new bootstrap.Modal(document.getElementById('scanning-modal'));
        myModal.show();
    }
});