import React, { Component } from 'react';

import classes from './ImageManager.module.css';
import Aux from '../../hoc/Aux'
import ImageControl from '../../component/Image/ImageControls/ImageControl';
import Modal from '../../component/UI/Modal/Modal';
import CompressProcess from '../../component/Image/CompressProcess/CompressProcess';
import axios from 'axios';
class ImageManager extends Component{
    state ={
        images : [
            

        ],
        active: false,
        indexCount:4,
        imageProcessing:false,
        currentImage: "",
        confirmedImage: "",
        currentCompressVal:1,
        confirmedCompressVal:1,
        loadingImage:false,
        shouldRenderImage:false
    }


    currentImageHandler = (event)=>{
        const filepath = event.target.value;
        //console.log('filepath',filepath);
        this.setState({currentImage: filepath, confirmedImage:filepath});
        //console.log('current image',this.state.currentImage);
    } 
    
    currentCompressValHandler = ( event ) =>{
        const compressVal = event.target.value;
        //console.log(compressVal)
        this.setState({currentCompressVal: compressVal});
        //console.log(this.state.currentCompressVal)
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

    //returns a JSON object with filesize before/after, filepath, etc
    

    //This will be a HTTP request for image compression later
    //Return: FILE SIZE- Before and After 
    compressAndShowImage = () =>{ 
        this.setState({loadingImage:true, confirmedCompressVal:this.state.currentCompressVal, 
            confirmedImage:this.state.currentImage })
        const images = [...this.state.images]
        const id = this.state.indexCount
        const filePath = this.state.confirmedImage;
        console.log('[ImageManger.js]',filePath)
        const compression = this.state.currentCompressVal;
        if(compression > 10){
            alert('Please pick a scaling factor between 1-10');
            return
        }

        const data = {
            filePath: filePath,
            compression:compression
        }

        axios.post('http://localhost:5000/compress', data).then(
            (request) =>{
                console.log(request);
                console.log(request.data)
                const fileSizeBefore = request.data.fileSizeBefore;
                const fileSizeAfter = request.data.fileSizeAfter;
                images.push({id:id, filePath:filePath, compression:compression,
                    sizeBefore:fileSizeBefore, sizeAfter:fileSizeAfter});
                
                this.setState((prevState, props)=>{
                    return{
                        images:images,
                        indexCount:prevState.indexCount + 1,
                        imageProcessing:!prevState.imageProcessing,
                        loadingImage:false
                    }
                })
                console.log(this.state.images) 
            }).catch(
                (error) => {
                    console.log(error)
                }
            )
        
    }

    render(){
        return(
            <Aux>
                <Modal
                loadingImage={this.state.loadingImage}
                show={this.state.imageProcessing}
                cancel={this.toggleImageModal}>
                    <CompressProcess
                    loadingImage={this.state.loadingImage}
                    currentImage={this.currentImageHandler}
                    cancel={this.toggleImageModal}
                    confirm={this.compressAndShowImage}
                    currentCompress={this.currentCompressValHandler}
                    ></CompressProcess>
                </Modal>
                <div className={classes.StartApp}>
                    <h1>START THE APPLICATION</h1>
                    <button onClick={this.toggleImageModal}>CLICK ME</button>
                </div>
                <ImageControl 
                images={this.state.images}
                compress = {this.state.confirmedCompressVal}
                click={this.removeImageHandler}
                shouldRenderImage = {this.state.shouldRenderImage}
                />
            </Aux>
        );
    }
}

export default ImageManager;
