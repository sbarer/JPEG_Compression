import React, { Component } from 'react';

import classes from './ImageControl.module.css';
import Image from '../Image'

class ImageControl extends Component{
    render(){
        console.log('[ImageControl.js]',this.props.compress)
        let images = null
        if(this.props.images===null){
            images=null
            console.log('[ImageControl.js] rendering images', images)
        }
        else{
                console.log('[ImageControl.js] rendering images', images)
                images = this.props.images.map((image, index)=>[
                <Image
                compress = {this.props.compress}
                sizeBefore={image.sizeBefore} 
                sizeAfter={image.sizeAfter}
                compression={image.compression}
                imagePath={image.filePath} 
                key={image.id}
                click={()=>this.props.click(index)}/>
           ]);
        }
        
        //console.log(images)
        return(
            <div className={classes.ImageControl}>
                {images}
            </div>
        )
    }
} 



export default ImageControl;