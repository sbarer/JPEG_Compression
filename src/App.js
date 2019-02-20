import React, { Component } from 'react';

import classes from  './App.module.css';
import Layout from './component/Layout/Layout'
import ImageManager from './containers/ImageManager/ImageManager';
class App extends Component {
  render() {
    return (
      <div className={classes.App}>
        <Layout>
          <ImageManager />
        </Layout>
      </div>
    );
  }
}

export default App;
