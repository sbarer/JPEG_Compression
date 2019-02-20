import React from 'react';

import classes from './Image.module.css'
import Aux from '../../hoc/Aux'
import Button from '../UI/Button/Button'
var images = require.context('../../assets/images', true);

class ImageD extends React.Component {
    render() {
        let img_src = images(`./${this.props.imagePath}`)
        return (
            <Aux>
            <div className={classes.Image}>
                <h1>{this.props.imagePath} {this.props.imagePath}</h1>
                <img width='300px' height='300px' src={img_src} alt=""/>
                <img width='300px' height='300px' src={img_src} alt=""/>
                <div className={classes.Filesize}>
                    <h1>File Size: </h1>
                    <Button
                    btnType='Danger'
                    clicked={this.props.click}
                    >REMOVE</Button>
                </div>

            </div>
            </Aux>
            
        );
    }
}

export default ImageD;