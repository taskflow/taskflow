export default {
  input: './index.js',

  // external dependencies
  external: [
    'd3'
  ],
  output: {
    file: 'bundle.js',
    format: 'iife',
    // global variable
    globals: {
      'd3': 'd3'
    }
  }
};
