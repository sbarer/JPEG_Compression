import React, { Component } from 'react';

import classes from './ImageManager.module.css';
import Aux from '../../hoc/Aux'
import ImageControl from '../../component/Image/ImageControls/ImageControl';

class ImageManager extends Component{
    state ={
        active: false,
        imageCount:0

    }

    render(){
        return(
            <Aux>
                <ImageControl />
            </Aux>
        );
    }
}

export default ImageManager;
