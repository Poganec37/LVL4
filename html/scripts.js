console.log('Учим'); console.log('js');

let r=t=y=7.43;

function doThis(){
	console.log(r+' '+t+' '+y);

	met:
		for(let num = 0; num<2;num++){
			for(let size =0;size<3;size++){
				if(size==2)break met;
				console.log(size);
			}
		}	
	
}

doThis();

function someSum(a,b){
	return a+b;
}
console.log(`Сумма: ${someSum(4,3)}`);
document.getElementById("tf").innerHTML = `Сумма: ${someSum(4,3)}`;

let box=`<b>Многострочный пробник</b>
<img src="img/logo.png"> `;

//document.getElementById("p1").innerHTML = "1 "+box;
//document.getElementById("p2").innerHTML = "2 "+box;

for (let i=1; i<=2;i++){
	document.getElementById("p"+i).innerHTML = i+" "+box;
}
