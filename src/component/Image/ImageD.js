import React from 'react';
var images = require.context('../../assets/images', true);

class ImageD extends React.Component {
    render() {
        let img_src = images(`./${this.props.imagePath}`)
        return (
            <img src={img_src} alt=""/>
        );
    }
}

export default ImageD;