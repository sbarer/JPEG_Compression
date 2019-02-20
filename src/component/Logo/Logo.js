import React from 'react';

import burgerLogo from '../../assets/images/JPEG.png';
import classes from './Logo.module.css';

const logo = ( props ) => (
    <div className={classes.Logo} >
        <img src={burgerLogo} alt='JPEG'></img>
    </div>
);

export default logo