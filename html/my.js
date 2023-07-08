var b = document.getElementById('one');
var H = document.getElementById('tx'); 
//наведение курсора
b.onmouseover = function (){
	H.innerHTML = "Гомер говорит: \'<b>Мужчина</b>\' ";
	this.style.backgroundImage = 'url(img/Hom1.png)';
	this.onclick = null;
}
//нажатие мыши- любая кнопка
b.onmousedown = function (){
	H.innerHTML = "Гомер говорит: \'<b>Женщина</b>\' ";
	this.style.backgroundImage = 'url(img/Hom2.png)';
	this.onclick = null;
}
//курсор уходит с элемента
b.onmouseout = function (){
	H.innerHTML ="";
	this.style.background = 'orange';
	this.onclick = null;
}
// движение
var sdvig = 0;
function gret() {
	if(sdvig<=200){document.getElementById('two').style.marginLeft = sdvig+'px';}
	sdvig = sdvig+4;
}
function move(){setInterval(gret, 50); sdvig = 0;}

console.log('Лабораторная работа по JS');

let r=t=y=7.43;
function someSum(a,b){
	return a+b;
}

function doThis(){
	console.log(r+' '+t+' '+y);
	document.getElementById("three").innerHTML += `4+3= ${someSum(4,3)} `;
	document.getElementById('three').innerHTML +='  Цикл FOR';
	for(let size =0;size<3;size++){
		console.log(size);
		document.getElementById('three').innerHTML += ' '+size;
	}	
	
}
doThis();