const fileInput=document.getElementById("fileInput"),
dropZone=document.getElementById("dropZone"),
workspace=document.getElementById("workspace"),
inputPreview=document.getElementById("inputPreview"),
inputSize=document.getElementById("inputSize"),
restoreButton=document.getElementById("restoreButton"),
downloadButton=document.getElementById("downloadButton"),
outputPreview=document.getElementById("outputPreview"),
outputPlaceholder=document.getElementById("outputPlaceholder"),
inferenceTime=document.getElementById("inferenceTime"),
metrics=document.getElementById("metrics"),
errorBox=document.getElementById("error");

let selectedFile=null;

fileInput.onchange=e=>{
    if(e.target.files.length) loadFile(e.target.files[0]);
};

dropZone.ondragover=e=>{
    e.preventDefault();
    dropZone.classList.add("dragging");
};

dropZone.ondragleave=()=>{
    dropZone.classList.remove("dragging");
};

dropZone.ondrop=e=>{
    e.preventDefault();
    dropZone.classList.remove("dragging");
    if(e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
};

async function loadFile(file){
    selectedFile=file;
    errorBox.classList.add("hidden");
    workspace.classList.remove("hidden");
    outputPreview.classList.add("hidden");
    outputPlaceholder.classList.remove("hidden");
    downloadButton.classList.add("hidden");
    metrics.classList.add("hidden");

    try{
        if(file.name.toLowerCase().endsWith(".npy"))
            await showNPY(file);
        else
            showImage(file);
    }catch(e){
        error(e.message);
    }
}

function showImage(file){
    const url=URL.createObjectURL(file);
    inputPreview.onload=()=>{
        inputSize.textContent=
            `${inputPreview.naturalWidth} × ${inputPreview.naturalHeight}`;
        inputPreview.classList.remove("hidden");
    };
    inputPreview.src=url;
}

async function showNPY(file){
    const b=await file.arrayBuffer(),n=parseNPY(b);
    const [h,w]=n.shape;

    inputSize.textContent=`${w} × ${h}`;

    const c=document.createElement("canvas");
    c.width=w;
    c.height=h;

    const ctx=c.getContext("2d");
    const img=ctx.createImageData(w,h);

    let min=Infinity,max=-Infinity;

    for(const v of n.data)
        if(Number.isFinite(v)){
            min=Math.min(min,v);
            max=Math.max(max,v);
        }

    const r=max-min||1;

    for(let i=0;i<n.data.length;i++){
        const v=Number.isFinite(n.data[i])
            ?(n.data[i]-min)/r:0;
        const p=Math.max(0,Math.min(255,Math.round(v*255)));
        const j=i*4;
        img.data[j]=p;
        img.data[j+1]=p;
        img.data[j+2]=p;
        img.data[j+3]=255;
    }

    ctx.putImageData(img,0,0);
    inputPreview.src=c.toDataURL("image/png");
    inputPreview.classList.remove("hidden");
}

function parseNPY(b){
    const x=new Uint8Array(b);

    if(x[0]!==147||x[1]!==78||x[2]!==85||
       x[3]!==77||x[4]!==80||x[5]!==89)
        throw Error("Invalid NPY file");

    const ver=x[6];
    const len=ver===1
        ?x[8]|x[9]<<8
        :new DataView(b).getUint32(8,true);

    const start=ver===1?10:12;

    const header=new TextDecoder()
        .decode(x.slice(start,start+len));

    const d=header.match(/'descr'\s*:\s*'([^']+)'/);
    const s=header.match(/'shape'\s*:\s*\(([^)]*)\)/);

    if(!d||!s) throw Error("Invalid NPY header");

    const shape=s[1].split(",")
        .map(v=>v.trim())
        .filter(Boolean)
        .map(Number);

    if(shape.length!==2)
        throw Error("NPY must be a 2D grayscale array");

    const count=shape[0]*shape[1];
    const pos=start+len;

    let data;

    if(d[1]==="<f4"||d[1]==="|f4")
        data=new Float32Array(b,pos,count);
    else if(d[1]==="<f8"||d[1]==="|f8")
        data=new Float64Array(b,pos,count);
    else if(d[1]==="|u1")
        data=new Uint8Array(b,pos,count);
    else
        throw Error("Unsupported NPY dtype: "+d[1]);

    return {data,shape};
}

restoreButton.onclick=async()=>{
    if(!selectedFile){
        error("Please select a file first.");
        return;
    }

    restoreButton.disabled=true;
    restoreButton.textContent="Restoring...";
    errorBox.classList.add("hidden");

    try{
        const form=new FormData();
        form.append("file",selectedFile);

        const t=performance.now();

        /* IMPORTANT: this is the API endpoint */
        const response=await fetch("/api/restore",{
            method:"POST",
            body:form
        });

        if(!response.ok)
            throw Error(await response.text());

        const blob=await response.blob();
        const url=URL.createObjectURL(blob);

        outputPreview.src=url;
        outputPreview.classList.remove("hidden");
        outputPlaceholder.classList.add("hidden");

        const serverTime=response.headers.get("X-Inference-Time");

        inferenceTime.textContent=serverTime
            ?`${Number(serverTime).toFixed(4)}s`
            :`${((performance.now()-t)/1000).toFixed(3)}s`;

        metrics.classList.remove("hidden");
        downloadButton.classList.remove("hidden");

        downloadButton.onclick=()=>{
            const a=document.createElement("a");
            a.href=url;
            a.download="SIV_AI_restored.png";
            a.click();
        };

    }catch(e){
        error(e.message);
    }finally{
        restoreButton.disabled=false;
        restoreButton.textContent="Restore with SIV-AI";
    }
};

function error(msg){
    errorBox.textContent=msg;
    errorBox.classList.remove("hidden");
}