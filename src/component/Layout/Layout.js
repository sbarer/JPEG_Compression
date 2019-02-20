import React from 'react';

import Aux from '../../hoc/Aux';


const layout = (props) =>{
    return(
    <Aux>
        <div>Logo, Menu, Back drop</div>
        <main>
            {props.children}
        </main>
    </Aux>
    )
}

export default layout;