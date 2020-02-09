//functions and utils
class Utils {
    constructor(...args) {
        //empty
    }
    sigmoid(x, derivative) {
        let fx = 1 / (1 + Math.exp(-x));    //Regular sigmoid function
        if (derivative)
           return fx * (1 - fx);            //When a derivative is given the function returns the reverse sigmoid function
        return fx;
    }
    random(m1, m2, l1, l2) {    //m1 & m2 represent layers next to each other
        //let m = m1 * m2;        //m1 & m2 are multiplied to get the total weights between layers
        let arr = [];
        for(var i=0;i<m1;i++) {
            let tmp = [];
            for(var j=0;j<m2;j++) {
                tmp.push((Math.random()*(l2+l1)) - l1);     //l2 is the upper limit whereas l1 is the lower limit*
            }                                               //l1 the *LL is always converted to a negative               
            arr.push(tmp);
        }
        return arr;
    }
    transpose(arr) {
        return arr[0].map((col, i) => arr.map(row => row[i]));      //Transposes any array ; Basically the array is split up in yx coordinates
    }
    matrix_from_num(num, sizeY, sizeX) {
        let matrix = []
        for(var i=0;i<sizeY;i++) {
            matrix[i] = [];                 //Simply creates a matrix from a single number for a given size
            for(var j=0;j<sizeX;j++) {
                matrix[i][j] = num;
            }
        }
        return matrix;
    }
    multiply(arr1, arr2) {
        let result = [];
        for (var i = 0; i < arr1.length; i++) {      //First array's length is used as reference for the wrapper loop
            result[i] = [];
            for (var j = 0; j < arr2[0].length; j++) {
                var sum = 0;
                for (var k = 0; k < arr1[0].length; k++) {
                    sum += arr1[i][k] * arr2[k][j];             //Relevant rows and cols are being multiplied
                }
                result[i][j] = sum;       //Adding the results of the calcs to the output result
            }
        }
        return result;
    }
    dotmap(matrix, fx, state) {
        fx = fx.toLowerCase();
        switch (fx) {
            case "sig" || "sigmoid":
                let product = [];
                matrix.forEach(element => {
                    product.push(element.map(v => utils.sigmoid(v, state)));
                });
                return product;
            default:
                return NaN;
        }
    }
    dotmultiply(arr1, arr2) {
        let result = [];
        for(var i=0;i<arr1.length;i++) {
            result[i] = [];
            for(var j=0;j<arr1[0].length;j++) {
                result[i][j] = arr1[i][j] * arr2[i][j];       //Multiplies the given arrays element wise
            }                                                  //(one on one) => Can only be applied to arrays
        }                                                      //with the same size.
        return result;
    }
    subtract(arr1, arr2) {
        let result = [];
        for(var i=0;i<arr1.length;i++) {
            result[i] = [];
            for(var j=0;j<arr1[0].length;j++) {
                result[i][j] = arr1[i][j] - arr2[i][j];       //Subtracts the given arrays element wise
            }                                                  //(one on one) => Can only be applied to arrays
        }                                                      //with the same size.
        return result;
    }
    add(arr1, arr2) {
        let result = [];
        for(var i=0;i<arr1.length;i++) {
            result[i] = [];
            for(var j=0;j<arr1[0].length;j++) {
                result[i][j] = arr1[i][j] + arr2[i][j];       //Adds the given arrays element wise
            }                                                  //(one on one) => Can only be applied to arrays
        }                                                      //with the same size.
        return result;
    }
    abs(matrix) {
        let product = [];
        matrix.forEach(element => {
            product.push(element.map(v => Math.abs(v)));
        });
        return product;
    }
    mean(matrix) {
        let product = 0;
        matrix.forEach(element => {
            element.forEach(el => {
                product += el;
            });
        });
        return product;
    }
}
const utils = new Utils;

class Neural_Network {
    constructor(...args) {
        this.input_nodes = args[0];
        this.hidden_nodes = args[1];
        this.output_nodes = args[2];

        this.epochs = 500000;
        this.lr = .5;
        this.output = 0;

        this.synapse0 = utils.random(this.input_nodes, this.hidden_nodes, 1, 1);
        this.synapse1 = utils.random(this.hidden_nodes, this.output_nodes, 1, 1);
    }
    train(input, target) {
        for(let i=0;i<this.epochs;i++) {
            //forward pass
            let input_layer = input;
            let hidden_layer = utils.multiply(input_layer, this.synapse0);
            hidden_layer = utils.dotmap(hidden_layer, "sig", false);
            let output_layer = utils.multiply(hidden_layer, this.synapse1);
            output_layer = utils.dotmap(output_layer, "sig", false);

            //backward pass
            let output_error = utils.subtract(target, output_layer);
            output_layer = utils.dotmap(output_layer, "sig", true);
            let output_delta = utils.dotmultiply(output_error, output_layer);
            let hidden_error = utils.multiply(output_delta, utils.transpose(this.synapse1));
            let hidden_delta = utils.dotmultiply(hidden_error, utils.dotmap(hidden_layer, "sig", true));
            //bis hier allet ok

            //gradient descent
            this.synapse1 = utils.add(this.synapse1, utils.multiply(utils.transpose(hidden_layer), utils.multiply(output_delta, utils.matrix_from_num(this.lr, 4, 1))));
            this.synapse0 = utils.add(this.synapse0, utils.multiply(utils.transpose(input_layer), utils.dotmultiply(hidden_delta, utils.matrix_from_num(this.lr, 4, 4))));
            this.output = output_layer;            

            if(i % 10000 == 0) {
                console.log(`Error: ${utils.mean(utils.abs(output_error))}`);
            }
            if(i == this.epochs-1) {
                console.log(`\nThe final error is: ${utils.mean(utils.abs(output_error))}`);
                console.log('\nFinal weights: Synapse0');
                console.log(this.synapse0);
                console.log('\nFinal weights: Synapse1');
                console.log(this.synapse1);
            }
        }
    }
    predict(input) {
        let input_layer = input;
        let hidden_layer = utils.multiply(input_layer, this.synapse0);
        hidden_layer = utils.dotmap(hidden_layer, "sig", false);
        let output_layer = utils.multiply(hidden_layer, this.synapse1);
        output_layer = utils.dotmap(output_layer, "sig", false);

        return output_layer;
    }
}
const input = [[0, 0], 
               [0, 1],
               [1, 0],
               [1, 1]];

const target = [[0], 
                [1], 
                [1], 
                [0]];

const Network = new Neural_Network(2, 4, 1);
Network.train(input, target);
console.log(Network.predict(input));