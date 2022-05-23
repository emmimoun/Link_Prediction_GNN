var acc = document.getElementsByClassName("accordion");
var panel =  document.getElementsByClassName("panel");


acc.addEventListener("click", function () {
    if (panel.style.display === "block") {
        panel.style.display = "none";
    } else {
        panel.style.display = "block";
    }
});
/*
                    allnewlinkArray=[];
                    for(let i in allnewlinkArray){
                       allnewlinkArray[i]=G2[i]['new_link'];
                    }
                    /*
                    newlinkArray = G2[0]['new_link'];
                    t=t+1;
                    for(let i in newlinkArray){
                       console.log(newlinkArray[i]);
                       let src = nodeArray2.find(({ name }) =>  name==newlinkArray[i][0]);
                       let trgt = nodeArray2.find(({ name }) =>  name==newlinkArray[i][1]);
                       console.log(src); console.log(trgt);
                       let linkobj={};
                       linkobj['source']=src;linkobj['target']=trgt;linkobj['weight']=t;
                       console.log(linkobj);
                       edgeArray2.push(linkobj);
                    }
                    console.log(newlinkArray)
                    console.log(nodeArray2);
                    console.log(edgeArray2);
                    
                    var t = 2; 
                    for(let i in allnewlinkArray){
                        newlinkArray = allnewlinkArray[i];
                        // completer ici ..........
                    }
                    */