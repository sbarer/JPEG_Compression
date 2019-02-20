import React, { Component } from 'react';

import classes from './ImageManager.module.css';
import Aux from '../../hoc/Aux'
import ImageControl from '../../component/Image/ImageControls/ImageControl';
import Modal from '../../component/UI/Modal/Modal';
import CompressProcess from '../../component/Image/CompressProcess/CompressProcess';
class ImageManager extends Component{
    state ={
        images : [
            {id:'1', filePath:'space.jpg', compression:1, sizeBefore:0, sizeAfter:0},
            {id:'2', filePath:'space.jpg', compression:2, sizeBefore:0, sizeAfter:0},
            {id:'3', filePath:'hotel_spot.jpg', compression:3, sizeBefore:0, sizeAfter:0}

        ],
        active: false,
        indexCount:4,
        imageProcessing:false,
        currentImage: "",
        currentCompressVal:1
    }


    currentImageHandler = (event)=>{
        const filepath = event.target.value;
        //console.log('filepath',filepath);
        this.setState({currentImage: filepath});
        //console.log('current image',this.state.currentImage);
    } 
    
    currentCompressValHandler = ( event ) =>{
        const compressVal = event.target.value;
        console.log(compressVal)
        this.setState({currentCompressVal: compressVal});
    }

    removeImageHandler = ( imageIndex ) =>{
        console.log('removing image')
        const images = [...this.state.images];
        images.splice(imageIndex, 1);
        this.setState({images:images})
        console.log(this.state.images)

    }

    toggleImageModal = () =>{
        this.setState({imageProcessing: !this.state.imageProcessing})
    }

    //This will be a HTTP request for image compression later
    //Return: FILE SIZE- Before and After 
    compressAndShowImage = () =>{
        //TODO: Backend request
        //fake for now...
        //expect to recieve 2 values fileSize Before Compression and fileSize after
        //JSON format 
        const images = [...this.state.images]
        const id = this.state.indexCount
        const filePath = this.state.currentImage;
        const compression = this.state.currentCompressVals
        const fileSizeBefore = 10;
        const fileSizeAfter = 5;
        images.push({id:id, filePath:filePath, compression:compression,
            sizeBefore:fileSizeBefore, sizeAfter:fileSizeAfter});
        
        this.setState((prevState, props)=>{
            return{
                images:images,
                indexCount:prevState.indexCount + 1,
                imageProcessing:!prevState.imageProcessing
            }
        })


        
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
                    confirm={this.compressAndShowImage}
                    currentCompress={this.currentCompressValHandler}
                    ></CompressProcess>
                </Modal>
                <div className={classes.StartApp}>
                    <h1>START OUR APPLICATION</h1>
                    <button onClick={this.toggleImageModal}>PRESS ME</button>
                </div>
                <ImageControl 
                images={this.state.images}
                click={this.removeImageHandler}
                />
            </Aux>
        );
    }
}

export default ImageManager;
