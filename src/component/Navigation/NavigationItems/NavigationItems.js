import React from 'react';

import classes from './NavigationItems.module.css';
import NavigationItem from './NavigationItem/NavigationItem';


const navigationItems = () =>(
    <ul className={classes.NavigationItems}>
        <NavigationItem link="/" active={true}>About</NavigationItem>
        <NavigationItem link="/" >JPEG</NavigationItem>
        <NavigationItem link="/" >DOCS</NavigationItem>
    </ul>
)

export default navigationItems;