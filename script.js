window.addEventListener('load', ()=>{ 
        
    resize(); // Resizes the canvas once the window loads 
    document.addEventListener('mousedown', startPainting); 
    document.addEventListener('mouseup', stopPainting); 
    document.addEventListener('mousemove', sketch); 
    window.addEventListener('resize', resize); 
}); 
    
const canvas = document.querySelector('#canvas'); 
   

const ctx = canvas.getContext('2d'); 
      

function resize(){ 
  ctx.canvas.width = 420;
  ctx.canvas.height = 420;

} 
    

let coord = {x:0 , y:0};  
   

let paint = false; 

function getPosition(event){ 
  coord.x = event.clientX - canvas.offsetLeft; 
  coord.y = event.clientY - canvas.offsetTop; 
} 
  

function startPainting(event){ 
  paint = true; 
  getPosition(event); 
} 
function stopPainting(){ 
  paint = false; 
} 
    
function sketch(event){ 
  if (!paint) return; 
  ctx.beginPath(); 
    
  ctx.lineWidth = 20; 
  ctx.lineCap = 'round'; 
    
  ctx.strokeStyle = 'black'; 
      
  ctx.moveTo(coord.x, coord.y); 
   
  getPosition(event); 
   
  ctx.lineTo(coord.x , coord.y); 

  ctx.stroke(); 
} 

const prediction = document.getElementById("prediction")

document.getElementById("senddata").addEventListener("click", () => {
    const canvas = document.getElementById("canvas")
    const result = canvas.getContext('2d').getImageData(0, 0, 420, 420)
    const resultarr = Array.from(result.data);

    const alpha_arr = []
  
    for (let i = 0; i < resultarr.length; i = i + 4) {
      alpha_arr.push(resultarr[i+3])
    }

    fetch('http://127.0.0.1:5000/', {
       method: 'POST',
       body: JSON.stringify({
         alpha_arr: alpha_arr
       })
     }).then(response => response.json())
     .then(response => {
      prediction.innerHTML = `Du hast eine ${response.prediction} gemalt, da bin ich mir zu ${parseFloat(response.prob * 100).toFixed(4)}% sicher`
         console.log(response)
      prediction.classList.remove("hidden")
      const context = canvas.getContext('2d');

      setTimeout(function(){ context.clearRect(0, 0, 420, 420); }, 1500);
      
         
     })
    .catch(err => {
      console.log(err);
    });
})