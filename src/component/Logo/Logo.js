import React from 'react';

import JPEGLogo from '../../assets/images/JPEG.png';
import classes from './Logo.module.css';

const logo = ( props ) => (
    <div className={classes.Logo} >
        <img src={JPEGLogo} alt='JPEG'></img>
    </div>
);

export default logo