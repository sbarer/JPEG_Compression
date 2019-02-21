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
        <footer className={classes.Footer}>
          <p>Copyright &copy;2018 Design by David Pham</p>
        </footer>
      </div>
    );
  }
}

export default App;
