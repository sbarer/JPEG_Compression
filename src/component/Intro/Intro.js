import React from 'react';

import classes from './Intro.module.css'

const intro = ( props ) => (
    <div className={classes.Intro}>
        <h1>Welcome to our JPEG Compression Application</h1>
        <p>We wrote this using React, Flask and PIL</p>
        <p>Follow us on Instagram</p>
        <p>Pick a color and get started</p>
        <p>Compressing your Files hard and fast</p>
    </div>
)

export default intro;