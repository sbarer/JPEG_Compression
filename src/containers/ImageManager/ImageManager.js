import React, { Component } from 'react';

import classes from './ImageManager.module.css';
import Aux from '../../hoc/Aux'
import ImageControl from '../../component/Image/ImageControls/ImageControl';
import Modal from '../../component/UI/Modal/Modal';
import CompressProcess from '../../component/Image/CompressProcess/CompressProcess';
class ImageManager extends Component{
    state ={
        images : [
            {id:'1', filePath:'space.jpg'},
            {id:'2', filePath:'space.jpg'},
            {id:'3', filePath:'hotel_spot.jpg'}

        ],
        active: false,
        indexCount:0,
        imageProcessing:false,
        currentImage: ""


    }

    currentImageHandler = (event)=>{
        const filepath = event.target.value
        console.log('filepath',filepath)
        this.setState({currentImage: filepath})
        console.log('current image',this.state.currentImage)
    }    
    toggleImageModal = () =>{
        this.setState({imageProcessing: !this.state.imageProcessing})
    }
    render(){
        return(
            <Aux>
                <Modal
                show={this.state.imageProcessing}
                cancel={this.toggleImageModal}>
                    <CompressProcess
                    currentImage={this.currentImageHandler}
                    cancel={this.toggleImageModal}
                    ></CompressProcess>
                </Modal>
                <div className={classes.StartApp}>
                    <h1>START OUR APPLICATION</h1>
                    <button onClick={this.toggleImageModal}>PRESS ME</button>
                </div>
                <ImageControl images={this.state.images}/>
            </Aux>
        );
    }
}

export default ImageManager;
