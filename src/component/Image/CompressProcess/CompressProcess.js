import React, {Component} from 'react';
import Aux from '../../../hoc/Aux';
import Button from '../../UI/Button/Button';

class CompressProcess extends Component{
    render(){
        let text = null;
        if(this.props.loadingImage){
            text = <h1>Loading Image...</h1>
        }
        else{
            text = (
                <div>
                    <h3>Hello</h3>
                    <p>Select the image you would like to compress</p>
                </div>
            )
        }
        return(
            <Aux>
                {text}
                <input type='text' onChange={this.props.currentImage}></input>
                <br />
                <p>Please select Quantization value (1-10)</p>
                <input type='text' onChange={this.props.currentCompress}></input>
                <br />
                <Button
                    clicked={this.props.cancel}
                    btnType='Danger'
                >CANCEL</Button>
                <Button
                    clicked={this.props.confirm}
                    btnType='Success'
                >CONTINUE</Button>
            </Aux>
        );
    }
}

export default CompressProcess;