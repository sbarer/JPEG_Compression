import React from 'react';

import classes from './Image.module.css'
import Aux from '../../hoc/Aux'
import Button from '../UI/Button/Button'
var images = require.context('../../assets/images', true);

class ImageD extends React.Component {

    state ={
        error:false,
        info:null
    }
    componentDidCatch(error, info){
        this.setState({
            error:error,
            info:info
        })
        console.log(this.state.info)
    }

    shouldComponentUpdate(){
        if(this.state.error){
            return false;
        } else {
            return true;
        }
    }
    render() {
        console.log('[Image.js]',this.props.compress)
        let img_src = images(`./${this.props.imagePath}`)
        let compressed_img = this.props.imagePath.split('.')
        let img_name = compressed_img[0]
        console.log(img_name)

        let compressed_img_src = String(img_name) + String(this.props.compress) + '.jpg'
        console.log('this is the compressed image name', compressed_img_src)
        let comp_img_src = images(`./${compressed_img_src}`)
        let compression_ratio = (this.props.sizeBefore/this.props.sizeAfter)

        compression_ratio = compression_ratio.toFixed(2)
        return (
            <div className={classes.Image}>
                <h1>{this.props.imagePath}</h1>
                    <img width='350px' height='350px' src={img_src} alt=""/>
                    <img width='350px' height='350px' src={comp_img_src} alt=""/> 
                <div className={classes.Filesize}>
                    <div className={classes.headers}>
                        <h2>File Size Before: {this.props.sizeBefore}kB</h2>
                    </div>
                    <div className={classes.headers}>
                        <h2>File Size After: {this.props.sizeAfter}kB</h2>
                    </div>
                    <div className={classes.headers}>
                        <h2>Quantization Scaling factor: {this.props.compression}</h2>
                    </div>
                    <div className={classes.headers}>
                        <h2>Compression Ratio: {compression_ratio}</h2>
                    </div>
                    
                    <Button
                    btnType='Danger'
                    clicked={this.props.click}
                    >REMOVE</Button>
                </div>

            </div>
            
        );
    }
}

export default ImageD;