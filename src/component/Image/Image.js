import React from 'react';

import classes from './Image.module.css'
import Aux from '../../hoc/Aux'
import Button from '../UI/Button/Button'
var images = require.context('../../assets/images', true);

class ImageD extends React.Component {
    render() {
        let img_src = images(`./${this.props.imagePath}`)
        let compression_ratio = (this.props.sizeBefore/this.props.sizeAfter)
        compression_ratio.toFixed(2)
        return (
            <div className={classes.Image}>
                <h1>{this.props.imagePath}</h1>
                <img width='350px' height='350px' src={img_src} alt=""/>
                <img width='350px' height='350px' src={img_src} alt=""/>
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