import React from 'react';

import classes from './Layout.module.css';
import Aux from '../../hoc/Aux';
import Toolbar from '../Navigation/Toolbar/Toolbar';
import Intro from '../Intro/Intro'

const layout = (props) =>{
    return(
    <Aux>
        <Toolbar />
        <div>
            <Intro />
        </div>
        <main className={classes.content}>
            {props.children}
        </main>
    </Aux>
    )
}

export default layout;