import React from 'react';

import classes from './ImageControl.module.css';
import Image from '../Image'

const imageControl = ( props ) => (
    <div className={classes.ImageControl}>
        <Image imagePath='space.jpg' />
        <Image imagePath='v_bridge.jpeg' />
        <Image imagePath='vancouver-coast.jpg' />
        <div>IMAGE 4</div>
    </div>
)

export default imageControl;