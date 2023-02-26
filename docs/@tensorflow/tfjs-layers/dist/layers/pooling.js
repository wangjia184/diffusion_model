/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * TensorFlow.js Layers: Pooling Layers.
 */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { imageDataFormat } from '../backend/common';
import * as K from '../backend/tfjs_backend';
import { checkDataFormat, checkPaddingMode, checkPoolMode } from '../common';
import { InputSpec } from '../engine/topology';
import { Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { convOutputLength } from '../utils/conv_utils';
import { assertPositiveInteger } from '../utils/generic_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
import { preprocessConv2DInput, preprocessConv3DInput } from './convolutional';
/**
 * 2D pooling.
 * @param x
 * @param poolSize
 * @param strides strides. Defaults to [1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 2D pooling.
 */
export function pool2d(x, poolSize, strides, padding, dataFormat, poolMode) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        checkPoolMode(poolMode);
        checkPaddingMode(padding);
        if (strides == null) {
            strides = [1, 1];
        }
        if (padding == null) {
            padding = 'valid';
        }
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        if (poolMode == null) {
            poolMode = 'max';
        }
        // TODO(cais): Remove the preprocessing step once deeplearn.js supports
        // dataFormat as an input argument.
        x = preprocessConv2DInput(x, dataFormat); // x is NHWC after preprocessing.
        let y;
        const paddingString = (padding === 'same') ? 'same' : 'valid';
        if (poolMode === 'max') {
            // TODO(cais): Rank check?
            y = tfc.maxPool(x, poolSize, strides, paddingString);
        }
        else { // 'avg'
            // TODO(cais): Check the dtype and rank of x and give clear error message
            //   if those are incorrect.
            y = tfc.avgPool(
            // TODO(cais): Rank check?
            x, poolSize, strides, paddingString);
        }
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 3, 1, 2]); // NHWC -> NCHW.
        }
        return y;
    });
}
/**
 * 3D pooling.
 * @param x
 * @param poolSize. Default to [1, 1, 1].
 * @param strides strides. Defaults to [1, 1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 3D pooling.
 */
export function pool3d(x, poolSize, strides, padding, dataFormat, poolMode) {
    return tidy(() => {
        checkDataFormat(dataFormat);
        checkPoolMode(poolMode);
        checkPaddingMode(padding);
        if (strides == null) {
            strides = [1, 1, 1];
        }
        if (padding == null) {
            padding = 'valid';
        }
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        if (poolMode == null) {
            poolMode = 'max';
        }
        // x is NDHWC after preprocessing.
        x = preprocessConv3DInput(x, dataFormat);
        let y;
        const paddingString = (padding === 'same') ? 'same' : 'valid';
        if (poolMode === 'max') {
            y = tfc.maxPool3d(x, poolSize, strides, paddingString);
        }
        else { // 'avg'
            y = tfc.avgPool3d(x, poolSize, strides, paddingString);
        }
        if (dataFormat === 'channelsFirst') {
            y = tfc.transpose(y, [0, 4, 1, 2, 3]); // NDHWC -> NCDHW.
        }
        return y;
    });
}
/**
 * Abstract class for different pooling 1D layers.
 */
export class Pooling1D extends Layer {
    /**
     *
     * @param args Parameters for the Pooling layer.
     *
     * config.poolSize defaults to 2.
     */
    constructor(args) {
        if (args.poolSize == null) {
            args.poolSize = 2;
        }
        super(args);
        if (typeof args.poolSize === 'number') {
            this.poolSize = [args.poolSize];
        }
        else if (Array.isArray(args.poolSize) &&
            args.poolSize.length === 1 &&
            typeof args.poolSize[0] === 'number') {
            this.poolSize = args.poolSize;
        }
        else {
            throw new ValueError(`poolSize for 1D convolutional layer must be a number or an ` +
                `Array of a single number, but received ` +
                `${JSON.stringify(args.poolSize)}`);
        }
        assertPositiveInteger(this.poolSize, 'poolSize');
        if (args.strides == null) {
            this.strides = this.poolSize;
        }
        else {
            if (typeof args.strides === 'number') {
                this.strides = [args.strides];
            }
            else if (Array.isArray(args.strides) &&
                args.strides.length === 1 &&
                typeof args.strides[0] === 'number') {
                this.strides = args.strides;
            }
            else {
                throw new ValueError(`strides for 1D convolutional layer must be a number or an ` +
                    `Array of a single number, but received ` +
                    `${JSON.stringify(args.strides)}`);
            }
        }
        assertPositiveInteger(this.strides, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        checkPaddingMode(this.padding);
        this.inputSpec = [new InputSpec({ ndim: 3 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const length = convOutputLength(inputShape[1], this.poolSize[0], this.padding, this.strides[0]);
        return [inputShape[0], length, inputShape[2]];
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            // Add dummy last dimension.
            inputs = K.expandDims(getExactlyOneTensor(inputs), 2);
            const output = this.poolingFunction(getExactlyOneTensor(inputs), [this.poolSize[0], 1], [this.strides[0], 1], this.padding, 'channelsLast');
            // Remove dummy last dimension.
            return tfc.squeeze(output, [2]);
        });
    }
    getConfig() {
        const config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class MaxPooling1D extends Pooling1D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
    }
}
/** @nocollapse */
MaxPooling1D.className = 'MaxPooling1D';
serialization.registerClass(MaxPooling1D);
export class AveragePooling1D extends Pooling1D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    }
}
/** @nocollapse */
AveragePooling1D.className = 'AveragePooling1D';
serialization.registerClass(AveragePooling1D);
/**
 * Abstract class for different pooling 2D layers.
 */
export class Pooling2D extends Layer {
    constructor(args) {
        if (args.poolSize == null) {
            args.poolSize = [2, 2];
        }
        super(args);
        this.poolSize = Array.isArray(args.poolSize) ?
            args.poolSize :
            [args.poolSize, args.poolSize];
        if (args.strides == null) {
            this.strides = this.poolSize;
        }
        else if (Array.isArray(args.strides)) {
            if (args.strides.length !== 2) {
                throw new ValueError(`If the strides property of a 2D pooling layer is an Array, ` +
                    `it is expected to have a length of 2, but received length ` +
                    `${args.strides.length}.`);
            }
            this.strides = args.strides;
        }
        else {
            // `config.strides` is a number.
            this.strides = [args.strides, args.strides];
        }
        assertPositiveInteger(this.poolSize, 'poolSize');
        assertPositiveInteger(this.strides, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        checkPaddingMode(this.padding);
        this.inputSpec = [new InputSpec({ ndim: 4 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        let rows = this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
        let cols = this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
        rows =
            convOutputLength(rows, this.poolSize[0], this.padding, this.strides[0]);
        cols =
            convOutputLength(cols, this.poolSize[1], this.padding, this.strides[1]);
        if (this.dataFormat === 'channelsFirst') {
            return [inputShape[0], inputShape[1], rows, cols];
        }
        else {
            return [inputShape[0], rows, cols, inputShape[3]];
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            return this.poolingFunction(getExactlyOneTensor(inputs), this.poolSize, this.strides, this.padding, this.dataFormat);
        });
    }
    getConfig() {
        const config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
            dataFormat: this.dataFormat
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class MaxPooling2D extends Pooling2D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
    }
}
/** @nocollapse */
MaxPooling2D.className = 'MaxPooling2D';
serialization.registerClass(MaxPooling2D);
export class AveragePooling2D extends Pooling2D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    }
}
/** @nocollapse */
AveragePooling2D.className = 'AveragePooling2D';
serialization.registerClass(AveragePooling2D);
/**
 * Abstract class for different pooling 3D layers.
 */
export class Pooling3D extends Layer {
    constructor(args) {
        if (args.poolSize == null) {
            args.poolSize = [2, 2, 2];
        }
        super(args);
        this.poolSize = Array.isArray(args.poolSize) ?
            args.poolSize :
            [args.poolSize, args.poolSize, args.poolSize];
        if (args.strides == null) {
            this.strides = this.poolSize;
        }
        else if (Array.isArray(args.strides)) {
            if (args.strides.length !== 3) {
                throw new ValueError(`If the strides property of a 3D pooling layer is an Array, ` +
                    `it is expected to have a length of 3, but received length ` +
                    `${args.strides.length}.`);
            }
            this.strides = args.strides;
        }
        else {
            // `config.strides` is a number.
            this.strides = [args.strides, args.strides, args.strides];
        }
        assertPositiveInteger(this.poolSize, 'poolSize');
        assertPositiveInteger(this.strides, 'strides');
        this.padding = args.padding == null ? 'valid' : args.padding;
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        checkPaddingMode(this.padding);
        this.inputSpec = [new InputSpec({ ndim: 5 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        let depths = this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
        let rows = this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
        let cols = this.dataFormat === 'channelsFirst' ? inputShape[4] : inputShape[3];
        depths = convOutputLength(depths, this.poolSize[0], this.padding, this.strides[0]);
        rows =
            convOutputLength(rows, this.poolSize[1], this.padding, this.strides[1]);
        cols =
            convOutputLength(cols, this.poolSize[2], this.padding, this.strides[2]);
        if (this.dataFormat === 'channelsFirst') {
            return [inputShape[0], inputShape[1], depths, rows, cols];
        }
        else {
            return [inputShape[0], depths, rows, cols, inputShape[4]];
        }
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            return this.poolingFunction(getExactlyOneTensor(inputs), this.poolSize, this.strides, this.padding, this.dataFormat);
        });
    }
    getConfig() {
        const config = {
            poolSize: this.poolSize,
            padding: this.padding,
            strides: this.strides,
            dataFormat: this.dataFormat
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class MaxPooling3D extends Pooling3D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool3d(inputs, poolSize, strides, padding, dataFormat, 'max');
    }
}
/** @nocollapse */
MaxPooling3D.className = 'MaxPooling3D';
serialization.registerClass(MaxPooling3D);
export class AveragePooling3D extends Pooling3D {
    constructor(args) {
        super(args);
    }
    poolingFunction(inputs, poolSize, strides, padding, dataFormat) {
        checkDataFormat(dataFormat);
        checkPaddingMode(padding);
        return pool3d(inputs, poolSize, strides, padding, dataFormat, 'avg');
    }
}
/** @nocollapse */
AveragePooling3D.className = 'AveragePooling3D';
serialization.registerClass(AveragePooling3D);
/**
 * Abstract class for different global pooling 1D layers.
 */
export class GlobalPooling1D extends Layer {
    constructor(args) {
        super(args);
        this.inputSpec = [new InputSpec({ ndim: 3 })];
    }
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[2]];
    }
    call(inputs, kwargs) {
        throw new NotImplementedError();
    }
}
export class GlobalAveragePooling1D extends GlobalPooling1D {
    constructor(args) {
        super(args || {});
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            return tfc.mean(input, 1);
        });
    }
}
/** @nocollapse */
GlobalAveragePooling1D.className = 'GlobalAveragePooling1D';
serialization.registerClass(GlobalAveragePooling1D);
export class GlobalMaxPooling1D extends GlobalPooling1D {
    constructor(args) {
        super(args || {});
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            return tfc.max(input, 1);
        });
    }
}
/** @nocollapse */
GlobalMaxPooling1D.className = 'GlobalMaxPooling1D';
serialization.registerClass(GlobalMaxPooling1D);
/**
 * Abstract class for different global pooling 2D layers.
 */
export class GlobalPooling2D extends Layer {
    constructor(args) {
        super(args);
        this.dataFormat =
            args.dataFormat == null ? 'channelsLast' : args.dataFormat;
        checkDataFormat(this.dataFormat);
        this.inputSpec = [new InputSpec({ ndim: 4 })];
    }
    computeOutputShape(inputShape) {
        inputShape = inputShape;
        if (this.dataFormat === 'channelsLast') {
            return [inputShape[0], inputShape[3]];
        }
        else {
            return [inputShape[0], inputShape[1]];
        }
    }
    call(inputs, kwargs) {
        throw new NotImplementedError();
    }
    getConfig() {
        const config = { dataFormat: this.dataFormat };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
export class GlobalAveragePooling2D extends GlobalPooling2D {
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            if (this.dataFormat === 'channelsLast') {
                return tfc.mean(input, [1, 2]);
            }
            else {
                return tfc.mean(input, [2, 3]);
            }
        });
    }
}
/** @nocollapse */
GlobalAveragePooling2D.className = 'GlobalAveragePooling2D';
serialization.registerClass(GlobalAveragePooling2D);
export class GlobalMaxPooling2D extends GlobalPooling2D {
    call(inputs, kwargs) {
        return tidy(() => {
            const input = getExactlyOneTensor(inputs);
            if (this.dataFormat === 'channelsLast') {
                return tfc.max(input, [1, 2]);
            }
            else {
                return tfc.max(input, [2, 3]);
            }
        });
    }
}
/** @nocollapse */
GlobalMaxPooling2D.className = 'GlobalMaxPooling2D';
serialization.registerClass(GlobalMaxPooling2D);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbGluZy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvcG9vbGluZy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOztHQUVHO0FBRUgsT0FBTyxLQUFLLEdBQUcsTUFBTSx1QkFBdUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsYUFBYSxFQUF3QyxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUVoRyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDbEQsT0FBTyxLQUFLLENBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsZUFBZSxFQUFFLGdCQUFnQixFQUFFLGFBQWEsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUMzRSxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDN0MsT0FBTyxFQUFDLEtBQUssRUFBWSxNQUFNLG9CQUFvQixDQUFDO0FBQ3BELE9BQU8sRUFBQyxtQkFBbUIsRUFBRSxVQUFVLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFHMUQsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDckQsT0FBTyxFQUFDLHFCQUFxQixFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDN0QsT0FBTyxFQUFDLGtCQUFrQixFQUFFLG1CQUFtQixFQUFDLE1BQU0sc0JBQXNCLENBQUM7QUFFN0UsT0FBTyxFQUFDLHFCQUFxQixFQUFFLHFCQUFxQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFN0U7Ozs7Ozs7OztHQVNHO0FBQ0gsTUFBTSxVQUFVLE1BQU0sQ0FDbEIsQ0FBUyxFQUFFLFFBQTBCLEVBQUUsT0FBMEIsRUFDakUsT0FBcUIsRUFBRSxVQUF1QixFQUM5QyxRQUFtQjtJQUNyQixPQUFPLElBQUksQ0FBQyxHQUFHLEVBQUU7UUFDZixlQUFlLENBQUMsVUFB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O0lBQ0wsQ0FBQzs7QUFaRCxrQkFBa0I7QUFDWCxnQ0FBUyxHQUFHLHdCQUF3QixDQUFDO0FBYTlDLGFBQWEsQ0FBQyxhQUFhLENBQUMsc0JBQXNCLENBQUMsQ0FBQztBQUVwRCxNQUFNLE9BQU8sa0JBQW1CLFNBQVEsZUFBZTtJQUk1QyxJQUFJLENBQUMsTUFBdUIsRUFBRSxNQUFjO1FBQ25ELE9BQU8sSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUNmLE1BQU0sS0FBSyxHQUFHLG1CQUFtQixDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQzFDLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxjQUFjLEVBQUU7Z0JBQ3RDLE9BQU8sR0FBRyxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMvQjtpQkFBTTtnQkFDTCxPQUFPLEdBQUcsQ0FBQyxHQUFHLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDL0I7UUFDSCxDQUFDLENBQUMsQ0FBQztJQUNMLENBQUM7O0FBWkQsa0JBQWtCO0FBQ1gsNEJBQVMsR0FBRyxvQkFBb0IsQ0FBQztBQWExQyxhQUFhLENBQUMsYUFBYSxDQUFDLGtCQUFrQixDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKipcbiAqIFRlbnNvckZsb3cuanMgTGF5ZXJzOiBQb29saW5nIExheWVycy5cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZmMgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7c2VyaWFsaXphdGlvbiwgVGVuc29yLCBUZW5zb3IzRCwgVGVuc29yNEQsIFRlbnNvcjVELCB0aWR5fSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2ltYWdlRGF0YUZvcm1hdH0gZnJvbSAnLi4vYmFja2VuZC9jb21tb24nO1xuaW1wb3J0ICogYXMgSyBmcm9tICcuLi9iYWNrZW5kL3RmanNfYmFja2VuZCc7XG5pbXBvcnQge2NoZWNrRGF0YUZvcm1hdCwgY2hlY2tQYWRkaW5nTW9kZSwgY2hlY2tQb29sTW9kZX0gZnJvbSAnLi4vY29tbW9uJztcbmltcG9ydCB7SW5wdXRTcGVjfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtMYXllciwgTGF5ZXJBcmdzfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtOb3RJbXBsZW1lbnRlZEVycm9yLCBWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtEYXRhRm9ybWF0LCBQYWRkaW5nTW9kZSwgUG9vbE1vZGUsIFNoYXBlfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvY29tbW9uJztcbmltcG9ydCB7S3dhcmdzfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2NvbnZPdXRwdXRMZW5ndGh9IGZyb20gJy4uL3V0aWxzL2NvbnZfdXRpbHMnO1xuaW1wb3J0IHthc3NlcnRQb3NpdGl2ZUludGVnZXJ9IGZyb20gJy4uL3V0aWxzL2dlbmVyaWNfdXRpbHMnO1xuaW1wb3J0IHtnZXRFeGFjdGx5T25lU2hhcGUsIGdldEV4YWN0bHlPbmVUZW5zb3J9IGZyb20gJy4uL3V0aWxzL3R5cGVzX3V0aWxzJztcblxuaW1wb3J0IHtwcmVwcm9jZXNzQ29udjJESW5wdXQsIHByZXByb2Nlc3NDb252M0RJbnB1dH0gZnJvbSAnLi9jb252b2x1dGlvbmFsJztcblxuLyoqXG4gKiAyRCBwb29saW5nLlxuICogQHBhcmFtIHhcbiAqIEBwYXJhbSBwb29sU2l6ZVxuICogQHBhcmFtIHN0cmlkZXMgc3RyaWRlcy4gRGVmYXVsdHMgdG8gWzEsIDFdLlxuICogQHBhcmFtIHBhZGRpbmcgcGFkZGluZy4gRGVmYXVsdHMgdG8gJ3ZhbGlkJy5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0IGRhdGEgZm9ybWF0LiBEZWZhdWx0cyB0byAnY2hhbm5lbHNMYXN0Jy5cbiAqIEBwYXJhbSBwb29sTW9kZSBNb2RlIG9mIHBvb2xpbmcuIERlZmF1bHRzIHRvICdtYXgnLlxuICogQHJldHVybnMgUmVzdWx0IG9mIHRoZSAyRCBwb29saW5nLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcG9vbDJkKFxuICAgIHg6IFRlbnNvciwgcG9vbFNpemU6IFtudW1iZXIsIG51bWJlcl0sIHN0cmlkZXM/OiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIHBhZGRpbmc/OiBQYWRkaW5nTW9kZSwgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQsXG4gICAgcG9vbE1vZGU/OiBQb29sTW9kZSk6IFRlbnNvciB7XG4gIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQb29sTW9kZShwb29sTW9kZSk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZShwYWRkaW5nKTtcbiAgICBpZiAoc3RyaWRlcyA9PSBudWxsKSB7XG4gICAgICBzdHJpZGVzID0gWzEsIDFdO1xuICAgIH1cbiAgICBpZiAocGFkZGluZyA9PSBudWxsKSB7XG4gICAgICBwYWRkaW5nID0gJ3ZhbGlkJztcbiAgICB9XG4gICAgaWYgKGRhdGFGb3JtYXQgPT0gbnVsbCkge1xuICAgICAgZGF0YUZvcm1hdCA9IGltYWdlRGF0YUZvcm1hdCgpO1xuICAgIH1cbiAgICBpZiAocG9vbE1vZGUgPT0gbnVsbCkge1xuICAgICAgcG9vbE1vZGUgPSAnbWF4JztcbiAgICB9XG5cbiAgICAvLyBUT0RPKGNhaXMpOiBSZW1vdmUgdGhlIHByZXByb2Nlc3Npbmcgc3RlcCBvbmNlIGRlZXBsZWFybi5qcyBzdXBwb3J0c1xuICAgIC8vIGRhdGFGb3JtYXQgYXMgYW4gaW5wdXQgYXJndW1lbnQuXG4gICAgeCA9IHByZXByb2Nlc3NDb252MkRJbnB1dCh4LCBkYXRhRm9ybWF0KTsgIC8vIHggaXMgTkhXQyBhZnRlciBwcmVwcm9jZXNzaW5nLlxuICAgIGxldCB5OiBUZW5zb3I7XG4gICAgY29uc3QgcGFkZGluZ1N0cmluZyA9IChwYWRkaW5nID09PSAnc2FtZScpID8gJ3NhbWUnIDogJ3ZhbGlkJztcbiAgICBpZiAocG9vbE1vZGUgPT09ICdtYXgnKSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBSYW5rIGNoZWNrP1xuICAgICAgeSA9IHRmYy5tYXhQb29sKHggYXMgVGVuc29yNEQsIHBvb2xTaXplLCBzdHJpZGVzLCBwYWRkaW5nU3RyaW5nKTtcbiAgICB9IGVsc2UgeyAgLy8gJ2F2ZydcbiAgICAgIC8vIFRPRE8oY2Fpcyk6IENoZWNrIHRoZSBkdHlwZSBhbmQgcmFuayBvZiB4IGFuZCBnaXZlIGNsZWFyIGVycm9yIG1lc3NhZ2VcbiAgICAgIC8vICAgaWYgdGhvc2UgYXJlIGluY29ycmVjdC5cbiAgICAgIHkgPSB0ZmMuYXZnUG9vbChcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBSYW5rIGNoZWNrP1xuICAgICAgICAgIHggYXMgVGVuc29yM0QgfCBUZW5zb3I0RCwgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmdTdHJpbmcpO1xuICAgIH1cbiAgICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICB5ID0gdGZjLnRyYW5zcG9zZSh5LCBbMCwgMywgMSwgMl0pOyAgLy8gTkhXQyAtPiBOQ0hXLlxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfSk7XG59XG5cbi8qKlxuICogM0QgcG9vbGluZy5cbiAqIEBwYXJhbSB4XG4gKiBAcGFyYW0gcG9vbFNpemUuIERlZmF1bHQgdG8gWzEsIDEsIDFdLlxuICogQHBhcmFtIHN0cmlkZXMgc3RyaWRlcy4gRGVmYXVsdHMgdG8gWzEsIDEsIDFdLlxuICogQHBhcmFtIHBhZGRpbmcgcGFkZGluZy4gRGVmYXVsdHMgdG8gJ3ZhbGlkJy5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0IGRhdGEgZm9ybWF0LiBEZWZhdWx0cyB0byAnY2hhbm5lbHNMYXN0Jy5cbiAqIEBwYXJhbSBwb29sTW9kZSBNb2RlIG9mIHBvb2xpbmcuIERlZmF1bHRzIHRvICdtYXgnLlxuICogQHJldHVybnMgUmVzdWx0IG9mIHRoZSAzRCBwb29saW5nLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcG9vbDNkKFxuICAgIHg6IFRlbnNvcjVELCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHN0cmlkZXM/OiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHBhZGRpbmc/OiBQYWRkaW5nTW9kZSxcbiAgICBkYXRhRm9ybWF0PzogRGF0YUZvcm1hdCwgcG9vbE1vZGU/OiBQb29sTW9kZSk6IFRlbnNvciB7XG4gIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQb29sTW9kZShwb29sTW9kZSk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZShwYWRkaW5nKTtcbiAgICBpZiAoc3RyaWRlcyA9PSBudWxsKSB7XG4gICAgICBzdHJpZGVzID0gWzEsIDEsIDFdO1xuICAgIH1cbiAgICBpZiAocGFkZGluZyA9PSBudWxsKSB7XG4gICAgICBwYWRkaW5nID0gJ3ZhbGlkJztcbiAgICB9XG4gICAgaWYgKGRhdGFGb3JtYXQgPT0gbnVsbCkge1xuICAgICAgZGF0YUZvcm1hdCA9IGltYWdlRGF0YUZvcm1hdCgpO1xuICAgIH1cbiAgICBpZiAocG9vbE1vZGUgPT0gbnVsbCkge1xuICAgICAgcG9vbE1vZGUgPSAnbWF4JztcbiAgICB9XG5cbiAgICAvLyB4IGlzIE5ESFdDIGFmdGVyIHByZXByb2Nlc3NpbmcuXG4gICAgeCA9IHByZXByb2Nlc3NDb252M0RJbnB1dCh4IGFzIFRlbnNvciwgZGF0YUZvcm1hdCkgYXMgVGVuc29yNUQ7XG4gICAgbGV0IHk6IFRlbnNvcjtcbiAgICBjb25zdCBwYWRkaW5nU3RyaW5nID0gKHBhZGRpbmcgPT09ICdzYW1lJykgPyAnc2FtZScgOiAndmFsaWQnO1xuICAgIGlmIChwb29sTW9kZSA9PT0gJ21heCcpIHtcbiAgICAgIHkgPSB0ZmMubWF4UG9vbDNkKHgsIHBvb2xTaXplLCBzdHJpZGVzLCBwYWRkaW5nU3RyaW5nKTtcbiAgICB9IGVsc2UgeyAgLy8gJ2F2ZydcbiAgICAgIHkgPSB0ZmMuYXZnUG9vbDNkKHgsIHBvb2xTaXplLCBzdHJpZGVzLCBwYWRkaW5nU3RyaW5nKTtcbiAgICB9XG4gICAgaWYgKGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgeSA9IHRmYy50cmFuc3Bvc2UoeSwgWzAsIDQsIDEsIDIsIDNdKTsgIC8vIE5ESFdDIC0+IE5DREhXLlxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfSk7XG59XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBQb29saW5nMURMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogU2l6ZSBvZiB0aGUgd2luZG93IHRvIHBvb2wgb3Zlciwgc2hvdWxkIGJlIGFuIGludGVnZXIuXG4gICAqL1xuICBwb29sU2l6ZT86IG51bWJlcnxbbnVtYmVyXTtcbiAgLyoqXG4gICAqIFBlcmlvZCBhdCB3aGljaCB0byBzYW1wbGUgdGhlIHBvb2xlZCB2YWx1ZXMuXG4gICAqXG4gICAqIElmIGBudWxsYCwgZGVmYXVsdHMgdG8gYHBvb2xTaXplYC5cbiAgICovXG4gIHN0cmlkZXM/OiBudW1iZXJ8W251bWJlcl07XG4gIC8qKiBIb3cgdG8gZmlsbCBpbiBkYXRhIHRoYXQncyBub3QgYW4gaW50ZWdlciBtdWx0aXBsZSBvZiBwb29sU2l6ZS4gKi9cbiAgcGFkZGluZz86IFBhZGRpbmdNb2RlO1xufVxuXG4vKipcbiAqIEFic3RyYWN0IGNsYXNzIGZvciBkaWZmZXJlbnQgcG9vbGluZyAxRCBsYXllcnMuXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBQb29saW5nMUQgZXh0ZW5kcyBMYXllciB7XG4gIHByb3RlY3RlZCByZWFkb25seSBwb29sU2l6ZTogW251bWJlcl07XG4gIHByb3RlY3RlZCByZWFkb25seSBzdHJpZGVzOiBbbnVtYmVyXTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHBhZGRpbmc6IFBhZGRpbmdNb2RlO1xuXG4gIC8qKlxuICAgKlxuICAgKiBAcGFyYW0gYXJncyBQYXJhbWV0ZXJzIGZvciB0aGUgUG9vbGluZyBsYXllci5cbiAgICpcbiAgICogY29uZmlnLnBvb2xTaXplIGRlZmF1bHRzIHRvIDIuXG4gICAqL1xuICBjb25zdHJ1Y3RvcihhcmdzOiBQb29saW5nMURMYXllckFyZ3MpIHtcbiAgICBpZiAoYXJncy5wb29sU2l6ZSA9PSBudWxsKSB7XG4gICAgICBhcmdzLnBvb2xTaXplID0gMjtcbiAgICB9XG4gICAgc3VwZXIoYXJncyk7XG4gICAgaWYgKHR5cGVvZiBhcmdzLnBvb2xTaXplID09PSAnbnVtYmVyJykge1xuICAgICAgdGhpcy5wb29sU2l6ZSA9IFthcmdzLnBvb2xTaXplXTtcbiAgICB9IGVsc2UgaWYgKFxuICAgICAgICBBcnJheS5pc0FycmF5KGFyZ3MucG9vbFNpemUpICYmXG4gICAgICAgIChhcmdzLnBvb2xTaXplIGFzIG51bWJlcltdKS5sZW5ndGggPT09IDEgJiZcbiAgICAgICAgdHlwZW9mIChhcmdzLnBvb2xTaXplIGFzIG51bWJlcltdKVswXSA9PT0gJ251bWJlcicpIHtcbiAgICAgIHRoaXMucG9vbFNpemUgPSBhcmdzLnBvb2xTaXplO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgcG9vbFNpemUgZm9yIDFEIGNvbnZvbHV0aW9uYWwgbGF5ZXIgbXVzdCBiZSBhIG51bWJlciBvciBhbiBgICtcbiAgICAgICAgICBgQXJyYXkgb2YgYSBzaW5nbGUgbnVtYmVyLCBidXQgcmVjZWl2ZWQgYCArXG4gICAgICAgICAgYCR7SlNPTi5zdHJpbmdpZnkoYXJncy5wb29sU2l6ZSl9YCk7XG4gICAgfVxuICAgIGFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLnBvb2xTaXplLCAncG9vbFNpemUnKTtcbiAgICBpZiAoYXJncy5zdHJpZGVzID09IG51bGwpIHtcbiAgICAgIHRoaXMuc3RyaWRlcyA9IHRoaXMucG9vbFNpemU7XG4gICAgfSBlbHNlIHtcbiAgICAgIGlmICh0eXBlb2YgYXJncy5zdHJpZGVzID09PSAnbnVtYmVyJykge1xuICAgICAgICB0aGlzLnN0cmlkZXMgPSBbYXJncy5zdHJpZGVzXTtcbiAgICAgIH0gZWxzZSBpZiAoXG4gICAgICAgICAgQXJyYXkuaXNBcnJheShhcmdzLnN0cmlkZXMpICYmXG4gICAgICAgICAgKGFyZ3Muc3RyaWRlcyBhcyBudW1iZXJbXSkubGVuZ3RoID09PSAxICYmXG4gICAgICAgICAgdHlwZW9mIChhcmdzLnN0cmlkZXMgYXMgbnVtYmVyW10pWzBdID09PSAnbnVtYmVyJykge1xuICAgICAgICB0aGlzLnN0cmlkZXMgPSBhcmdzLnN0cmlkZXM7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgIGBzdHJpZGVzIGZvciAxRCBjb252b2x1dGlvbmFsIGxheWVyIG11c3QgYmUgYSBudW1iZXIgb3IgYW4gYCArXG4gICAgICAgICAgICBgQXJyYXkgb2YgYSBzaW5nbGUgbnVtYmVyLCBidXQgcmVjZWl2ZWQgYCArXG4gICAgICAgICAgICBgJHtKU09OLnN0cmluZ2lmeShhcmdzLnN0cmlkZXMpfWApO1xuICAgICAgfVxuICAgIH1cbiAgICBhc3NlcnRQb3NpdGl2ZUludGVnZXIodGhpcy5zdHJpZGVzLCAnc3RyaWRlcycpO1xuXG4gICAgdGhpcy5wYWRkaW5nID0gYXJncy5wYWRkaW5nID09IG51bGwgPyAndmFsaWQnIDogYXJncy5wYWRkaW5nO1xuICAgIGNoZWNrUGFkZGluZ01vZGUodGhpcy5wYWRkaW5nKTtcbiAgICB0aGlzLmlucHV0U3BlYyA9IFtuZXcgSW5wdXRTcGVjKHtuZGltOiAzfSldO1xuICB9XG5cbiAgb3ZlcnJpZGUgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IGxlbmd0aCA9IGNvbnZPdXRwdXRMZW5ndGgoXG4gICAgICAgIGlucHV0U2hhcGVbMV0sIHRoaXMucG9vbFNpemVbMF0sIHRoaXMucGFkZGluZywgdGhpcy5zdHJpZGVzWzBdKTtcbiAgICByZXR1cm4gW2lucHV0U2hhcGVbMF0sIGxlbmd0aCwgaW5wdXRTaGFwZVsyXV07XG4gIH1cblxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgcGFkZGluZzogUGFkZGluZ01vZGUsIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3I7XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIHRoaXMuaW52b2tlQ2FsbEhvb2soaW5wdXRzLCBrd2FyZ3MpO1xuICAgICAgLy8gQWRkIGR1bW15IGxhc3QgZGltZW5zaW9uLlxuICAgICAgaW5wdXRzID0gSy5leHBhbmREaW1zKGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKSwgMik7XG4gICAgICBjb25zdCBvdXRwdXQgPSB0aGlzLnBvb2xpbmdGdW5jdGlvbihcbiAgICAgICAgICBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyksIFt0aGlzLnBvb2xTaXplWzBdLCAxXSxcbiAgICAgICAgICBbdGhpcy5zdHJpZGVzWzBdLCAxXSwgdGhpcy5wYWRkaW5nLCAnY2hhbm5lbHNMYXN0Jyk7XG4gICAgICAvLyBSZW1vdmUgZHVtbXkgbGFzdCBkaW1lbnNpb24uXG4gICAgICByZXR1cm4gdGZjLnNxdWVlemUob3V0cHV0LCBbMl0pO1xuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnID0ge1xuICAgICAgcG9vbFNpemU6IHRoaXMucG9vbFNpemUsXG4gICAgICBwYWRkaW5nOiB0aGlzLnBhZGRpbmcsXG4gICAgICBzdHJpZGVzOiB0aGlzLnN0cmlkZXMsXG4gICAgfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIE1heFBvb2xpbmcxRCBleHRlbmRzIFBvb2xpbmcxRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ01heFBvb2xpbmcxRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmcxRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXSwgc3RyaWRlczogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHBhZGRpbmc6IFBhZGRpbmdNb2RlLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yIHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZShwYWRkaW5nKTtcbiAgICByZXR1cm4gcG9vbDJkKGlucHV0cywgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsICdtYXgnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1heFBvb2xpbmcxRCk7XG5cbmV4cG9ydCBjbGFzcyBBdmVyYWdlUG9vbGluZzFEIGV4dGVuZHMgUG9vbGluZzFEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQXZlcmFnZVBvb2xpbmcxRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmcxRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXSwgc3RyaWRlczogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHBhZGRpbmc6IFBhZGRpbmdNb2RlLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yIHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZShwYWRkaW5nKTtcbiAgICByZXR1cm4gcG9vbDJkKGlucHV0cywgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsICdhdmcnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEF2ZXJhZ2VQb29saW5nMUQpO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgUG9vbGluZzJETGF5ZXJBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEZhY3RvcnMgYnkgd2hpY2ggdG8gZG93bnNjYWxlIGluIGVhY2ggZGltZW5zaW9uIFt2ZXJ0aWNhbCwgaG9yaXpvbnRhbF0uXG4gICAqIEV4cGVjdHMgYW4gaW50ZWdlciBvciBhbiBhcnJheSBvZiAyIGludGVnZXJzLlxuICAgKlxuICAgKiBGb3IgZXhhbXBsZSwgYFsyLCAyXWAgd2lsbCBoYWx2ZSB0aGUgaW5wdXQgaW4gYm90aCBzcGF0aWFsIGRpbWVuc2lvbnMuXG4gICAqIElmIG9ubHkgb25lIGludGVnZXIgaXMgc3BlY2lmaWVkLCB0aGUgc2FtZSB3aW5kb3cgbGVuZ3RoXG4gICAqIHdpbGwgYmUgdXNlZCBmb3IgYm90aCBkaW1lbnNpb25zLlxuICAgKi9cbiAgcG9vbFNpemU/OiBudW1iZXJ8W251bWJlciwgbnVtYmVyXTtcblxuICAvKipcbiAgICogVGhlIHNpemUgb2YgdGhlIHN0cmlkZSBpbiBlYWNoIGRpbWVuc2lvbiBvZiB0aGUgcG9vbGluZyB3aW5kb3cuIEV4cGVjdHNcbiAgICogYW4gaW50ZWdlciBvciBhbiBhcnJheSBvZiAyIGludGVnZXJzLiBJbnRlZ2VyLCB0dXBsZSBvZiAyIGludGVnZXJzLCBvclxuICAgKiBOb25lLlxuICAgKlxuICAgKiBJZiBgbnVsbGAsIGRlZmF1bHRzIHRvIGBwb29sU2l6ZWAuXG4gICAqL1xuICBzdHJpZGVzPzogbnVtYmVyfFtudW1iZXIsIG51bWJlcl07XG5cbiAgLyoqIFRoZSBwYWRkaW5nIHR5cGUgdG8gdXNlIGZvciB0aGUgcG9vbGluZyBsYXllci4gKi9cbiAgcGFkZGluZz86IFBhZGRpbmdNb2RlO1xuICAvKiogVGhlIGRhdGEgZm9ybWF0IHRvIHVzZSBmb3IgdGhlIHBvb2xpbmcgbGF5ZXIuICovXG4gIGRhdGFGb3JtYXQ/OiBEYXRhRm9ybWF0O1xufVxuXG4vKipcbiAqIEFic3RyYWN0IGNsYXNzIGZvciBkaWZmZXJlbnQgcG9vbGluZyAyRCBsYXllcnMuXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBQb29saW5nMkQgZXh0ZW5kcyBMYXllciB7XG4gIHByb3RlY3RlZCByZWFkb25seSBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl07XG4gIHByb3RlY3RlZCByZWFkb25seSBwYWRkaW5nOiBQYWRkaW5nTW9kZTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQ7XG5cbiAgY29uc3RydWN0b3IoYXJnczogUG9vbGluZzJETGF5ZXJBcmdzKSB7XG4gICAgaWYgKGFyZ3MucG9vbFNpemUgPT0gbnVsbCkge1xuICAgICAgYXJncy5wb29sU2l6ZSA9IFsyLCAyXTtcbiAgICB9XG4gICAgc3VwZXIoYXJncyk7XG4gICAgdGhpcy5wb29sU2l6ZSA9IEFycmF5LmlzQXJyYXkoYXJncy5wb29sU2l6ZSkgP1xuICAgICAgICBhcmdzLnBvb2xTaXplIDpcbiAgICAgICAgW2FyZ3MucG9vbFNpemUsIGFyZ3MucG9vbFNpemVdO1xuICAgIGlmIChhcmdzLnN0cmlkZXMgPT0gbnVsbCkge1xuICAgICAgdGhpcy5zdHJpZGVzID0gdGhpcy5wb29sU2l6ZTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkoYXJncy5zdHJpZGVzKSkge1xuICAgICAgaWYgKGFyZ3Muc3RyaWRlcy5sZW5ndGggIT09IDIpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgSWYgdGhlIHN0cmlkZXMgcHJvcGVydHkgb2YgYSAyRCBwb29saW5nIGxheWVyIGlzIGFuIEFycmF5LCBgICtcbiAgICAgICAgICAgIGBpdCBpcyBleHBlY3RlZCB0byBoYXZlIGEgbGVuZ3RoIG9mIDIsIGJ1dCByZWNlaXZlZCBsZW5ndGggYCArXG4gICAgICAgICAgICBgJHthcmdzLnN0cmlkZXMubGVuZ3RofS5gKTtcbiAgICAgIH1cbiAgICAgIHRoaXMuc3RyaWRlcyA9IGFyZ3Muc3RyaWRlcztcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gYGNvbmZpZy5zdHJpZGVzYCBpcyBhIG51bWJlci5cbiAgICAgIHRoaXMuc3RyaWRlcyA9IFthcmdzLnN0cmlkZXMsIGFyZ3Muc3RyaWRlc107XG4gICAgfVxuICAgIGFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLnBvb2xTaXplLCAncG9vbFNpemUnKTtcbiAgICBhc3NlcnRQb3NpdGl2ZUludGVnZXIodGhpcy5zdHJpZGVzLCAnc3RyaWRlcycpO1xuICAgIHRoaXMucGFkZGluZyA9IGFyZ3MucGFkZGluZyA9PSBudWxsID8gJ3ZhbGlkJyA6IGFyZ3MucGFkZGluZztcbiAgICB0aGlzLmRhdGFGb3JtYXQgPVxuICAgICAgICBhcmdzLmRhdGFGb3JtYXQgPT0gbnVsbCA/ICdjaGFubmVsc0xhc3QnIDogYXJncy5kYXRhRm9ybWF0O1xuICAgIGNoZWNrRGF0YUZvcm1hdCh0aGlzLmRhdGFGb3JtYXQpO1xuICAgIGNoZWNrUGFkZGluZ01vZGUodGhpcy5wYWRkaW5nKTtcblxuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDR9KV07XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgbGV0IHJvd3MgPVxuICAgICAgICB0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0JyA/IGlucHV0U2hhcGVbMl0gOiBpbnB1dFNoYXBlWzFdO1xuICAgIGxldCBjb2xzID1cbiAgICAgICAgdGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcgPyBpbnB1dFNoYXBlWzNdIDogaW5wdXRTaGFwZVsyXTtcbiAgICByb3dzID1cbiAgICAgICAgY29udk91dHB1dExlbmd0aChyb3dzLCB0aGlzLnBvb2xTaXplWzBdLCB0aGlzLnBhZGRpbmcsIHRoaXMuc3RyaWRlc1swXSk7XG4gICAgY29scyA9XG4gICAgICAgIGNvbnZPdXRwdXRMZW5ndGgoY29scywgdGhpcy5wb29sU2l6ZVsxXSwgdGhpcy5wYWRkaW5nLCB0aGlzLnN0cmlkZXNbMV0pO1xuICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCBpbnB1dFNoYXBlWzFdLCByb3dzLCBjb2xzXTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCByb3dzLCBjb2xzLCBpbnB1dFNoYXBlWzNdXTtcbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgcGFkZGluZzogUGFkZGluZ01vZGUsIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQpOiBUZW5zb3I7XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIHRoaXMuaW52b2tlQ2FsbEhvb2soaW5wdXRzLCBrd2FyZ3MpO1xuICAgICAgcmV0dXJuIHRoaXMucG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgICAgIGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKSwgdGhpcy5wb29sU2l6ZSwgdGhpcy5zdHJpZGVzLFxuICAgICAgICAgIHRoaXMucGFkZGluZywgdGhpcy5kYXRhRm9ybWF0KTtcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZyA9IHtcbiAgICAgIHBvb2xTaXplOiB0aGlzLnBvb2xTaXplLFxuICAgICAgcGFkZGluZzogdGhpcy5wYWRkaW5nLFxuICAgICAgc3RyaWRlczogdGhpcy5zdHJpZGVzLFxuICAgICAgZGF0YUZvcm1hdDogdGhpcy5kYXRhRm9ybWF0XG4gICAgfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIE1heFBvb2xpbmcyRCBleHRlbmRzIFBvb2xpbmcyRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ01heFBvb2xpbmcyRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmcyRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXSwgc3RyaWRlczogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHBhZGRpbmc6IFBhZGRpbmdNb2RlLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yIHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZShwYWRkaW5nKTtcbiAgICByZXR1cm4gcG9vbDJkKGlucHV0cywgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsICdtYXgnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1heFBvb2xpbmcyRCk7XG5cbmV4cG9ydCBjbGFzcyBBdmVyYWdlUG9vbGluZzJEIGV4dGVuZHMgUG9vbGluZzJEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQXZlcmFnZVBvb2xpbmcyRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFBvb2xpbmcyRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHBvb2xpbmdGdW5jdGlvbihcbiAgICAgIGlucHV0czogVGVuc29yLCBwb29sU2l6ZTogW251bWJlciwgbnVtYmVyXSwgc3RyaWRlczogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHBhZGRpbmc6IFBhZGRpbmdNb2RlLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0KTogVGVuc29yIHtcbiAgICBjaGVja0RhdGFGb3JtYXQoZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZShwYWRkaW5nKTtcbiAgICByZXR1cm4gcG9vbDJkKGlucHV0cywgcG9vbFNpemUsIHN0cmlkZXMsIHBhZGRpbmcsIGRhdGFGb3JtYXQsICdhdmcnKTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEF2ZXJhZ2VQb29saW5nMkQpO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgUG9vbGluZzNETGF5ZXJBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEZhY3RvcnMgYnkgd2hpY2ggdG8gZG93bnNjYWxlIGluIGVhY2ggZGltZW5zaW9uIFtkZXB0aCwgaGVpZ2h0LCB3aWR0aF0uXG4gICAqIEV4cGVjdHMgYW4gaW50ZWdlciBvciBhbiBhcnJheSBvZiAzIGludGVnZXJzLlxuICAgKlxuICAgKiBGb3IgZXhhbXBsZSwgYFsyLCAyLCAyXWAgd2lsbCBoYWx2ZSB0aGUgaW5wdXQgaW4gdGhyZWUgZGltZW5zaW9ucy5cbiAgICogSWYgb25seSBvbmUgaW50ZWdlciBpcyBzcGVjaWZpZWQsIHRoZSBzYW1lIHdpbmRvdyBsZW5ndGhcbiAgICogd2lsbCBiZSB1c2VkIGZvciBhbGwgZGltZW5zaW9ucy5cbiAgICovXG4gIHBvb2xTaXplPzogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcblxuICAvKipcbiAgICogVGhlIHNpemUgb2YgdGhlIHN0cmlkZSBpbiBlYWNoIGRpbWVuc2lvbiBvZiB0aGUgcG9vbGluZyB3aW5kb3cuIEV4cGVjdHNcbiAgICogYW4gaW50ZWdlciBvciBhbiBhcnJheSBvZiAzIGludGVnZXJzLiBJbnRlZ2VyLCB0dXBsZSBvZiAzIGludGVnZXJzLCBvclxuICAgKiBOb25lLlxuICAgKlxuICAgKiBJZiBgbnVsbGAsIGRlZmF1bHRzIHRvIGBwb29sU2l6ZWAuXG4gICAqL1xuICBzdHJpZGVzPzogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcblxuICAvKiogVGhlIHBhZGRpbmcgdHlwZSB0byB1c2UgZm9yIHRoZSBwb29saW5nIGxheWVyLiAqL1xuICBwYWRkaW5nPzogUGFkZGluZ01vZGU7XG4gIC8qKiBUaGUgZGF0YSBmb3JtYXQgdG8gdXNlIGZvciB0aGUgcG9vbGluZyBsYXllci4gKi9cbiAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQ7XG59XG5cbi8qKlxuICogQWJzdHJhY3QgY2xhc3MgZm9yIGRpZmZlcmVudCBwb29saW5nIDNEIGxheWVycy5cbiAqL1xuZXhwb3J0IGFic3RyYWN0IGNsYXNzIFBvb2xpbmczRCBleHRlbmRzIExheWVyIHtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHByb3RlY3RlZCByZWFkb25seSBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHByb3RlY3RlZCByZWFkb25seSBwYWRkaW5nOiBQYWRkaW5nTW9kZTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQ7XG5cbiAgY29uc3RydWN0b3IoYXJnczogUG9vbGluZzNETGF5ZXJBcmdzKSB7XG4gICAgaWYgKGFyZ3MucG9vbFNpemUgPT0gbnVsbCkge1xuICAgICAgYXJncy5wb29sU2l6ZSA9IFsyLCAyLCAyXTtcbiAgICB9XG4gICAgc3VwZXIoYXJncyk7XG4gICAgdGhpcy5wb29sU2l6ZSA9IEFycmF5LmlzQXJyYXkoYXJncy5wb29sU2l6ZSkgP1xuICAgICAgICBhcmdzLnBvb2xTaXplIDpcbiAgICAgICAgW2FyZ3MucG9vbFNpemUsIGFyZ3MucG9vbFNpemUsIGFyZ3MucG9vbFNpemVdO1xuICAgIGlmIChhcmdzLnN0cmlkZXMgPT0gbnVsbCkge1xuICAgICAgdGhpcy5zdHJpZGVzID0gdGhpcy5wb29sU2l6ZTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkoYXJncy5zdHJpZGVzKSkge1xuICAgICAgaWYgKGFyZ3Muc3RyaWRlcy5sZW5ndGggIT09IDMpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgSWYgdGhlIHN0cmlkZXMgcHJvcGVydHkgb2YgYSAzRCBwb29saW5nIGxheWVyIGlzIGFuIEFycmF5LCBgICtcbiAgICAgICAgICAgIGBpdCBpcyBleHBlY3RlZCB0byBoYXZlIGEgbGVuZ3RoIG9mIDMsIGJ1dCByZWNlaXZlZCBsZW5ndGggYCArXG4gICAgICAgICAgICBgJHthcmdzLnN0cmlkZXMubGVuZ3RofS5gKTtcbiAgICAgIH1cbiAgICAgIHRoaXMuc3RyaWRlcyA9IGFyZ3Muc3RyaWRlcztcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gYGNvbmZpZy5zdHJpZGVzYCBpcyBhIG51bWJlci5cbiAgICAgIHRoaXMuc3RyaWRlcyA9IFthcmdzLnN0cmlkZXMsIGFyZ3Muc3RyaWRlcywgYXJncy5zdHJpZGVzXTtcbiAgICB9XG4gICAgYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMucG9vbFNpemUsICdwb29sU2l6ZScpO1xuICAgIGFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLnN0cmlkZXMsICdzdHJpZGVzJyk7XG4gICAgdGhpcy5wYWRkaW5nID0gYXJncy5wYWRkaW5nID09IG51bGwgPyAndmFsaWQnIDogYXJncy5wYWRkaW5nO1xuICAgIHRoaXMuZGF0YUZvcm1hdCA9XG4gICAgICAgIGFyZ3MuZGF0YUZvcm1hdCA9PSBudWxsID8gJ2NoYW5uZWxzTGFzdCcgOiBhcmdzLmRhdGFGb3JtYXQ7XG4gICAgY2hlY2tEYXRhRm9ybWF0KHRoaXMuZGF0YUZvcm1hdCk7XG4gICAgY2hlY2tQYWRkaW5nTW9kZSh0aGlzLnBhZGRpbmcpO1xuXG4gICAgdGhpcy5pbnB1dFNwZWMgPSBbbmV3IElucHV0U3BlYyh7bmRpbTogNX0pXTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBsZXQgZGVwdGhzID1cbiAgICAgICAgdGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcgPyBpbnB1dFNoYXBlWzJdIDogaW5wdXRTaGFwZVsxXTtcbiAgICBsZXQgcm93cyA9XG4gICAgICAgIHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnID8gaW5wdXRTaGFwZVszXSA6IGlucHV0U2hhcGVbMl07XG4gICAgbGV0IGNvbHMgPVxuICAgICAgICB0aGlzLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0JyA/IGlucHV0U2hhcGVbNF0gOiBpbnB1dFNoYXBlWzNdO1xuICAgIGRlcHRocyA9IGNvbnZPdXRwdXRMZW5ndGgoXG4gICAgICAgIGRlcHRocywgdGhpcy5wb29sU2l6ZVswXSwgdGhpcy5wYWRkaW5nLCB0aGlzLnN0cmlkZXNbMF0pO1xuICAgIHJvd3MgPVxuICAgICAgICBjb252T3V0cHV0TGVuZ3RoKHJvd3MsIHRoaXMucG9vbFNpemVbMV0sIHRoaXMucGFkZGluZywgdGhpcy5zdHJpZGVzWzFdKTtcbiAgICBjb2xzID1cbiAgICAgICAgY29udk91dHB1dExlbmd0aChjb2xzLCB0aGlzLnBvb2xTaXplWzJdLCB0aGlzLnBhZGRpbmcsIHRoaXMuc3RyaWRlc1syXSk7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICByZXR1cm4gW2lucHV0U2hhcGVbMF0sIGlucHV0U2hhcGVbMV0sIGRlcHRocywgcm93cywgY29sc107XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgZGVwdGhzLCByb3dzLCBjb2xzLCBpbnB1dFNoYXBlWzRdXTtcbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHBhZGRpbmc6IFBhZGRpbmdNb2RlLFxuICAgICAgZGF0YUZvcm1hdDogRGF0YUZvcm1hdCk6IFRlbnNvcjtcblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgdGhpcy5pbnZva2VDYWxsSG9vayhpbnB1dHMsIGt3YXJncyk7XG4gICAgICByZXR1cm4gdGhpcy5wb29saW5nRnVuY3Rpb24oXG4gICAgICAgICAgZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpLCB0aGlzLnBvb2xTaXplLCB0aGlzLnN0cmlkZXMsXG4gICAgICAgICAgdGhpcy5wYWRkaW5nLCB0aGlzLmRhdGFGb3JtYXQpO1xuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnID0ge1xuICAgICAgcG9vbFNpemU6IHRoaXMucG9vbFNpemUsXG4gICAgICBwYWRkaW5nOiB0aGlzLnBhZGRpbmcsXG4gICAgICBzdHJpZGVzOiB0aGlzLnN0cmlkZXMsXG4gICAgICBkYXRhRm9ybWF0OiB0aGlzLmRhdGFGb3JtYXRcbiAgICB9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgTWF4UG9vbGluZzNEIGV4dGVuZHMgUG9vbGluZzNEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTWF4UG9vbGluZzNEJztcbiAgY29uc3RydWN0b3IoYXJnczogUG9vbGluZzNETGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHBhZGRpbmc6IFBhZGRpbmdNb2RlLFxuICAgICAgZGF0YUZvcm1hdDogRGF0YUZvcm1hdCk6IFRlbnNvciB7XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGNoZWNrUGFkZGluZ01vZGUocGFkZGluZyk7XG4gICAgcmV0dXJuIHBvb2wzZChcbiAgICAgICAgaW5wdXRzIGFzIFRlbnNvcjVELCBwb29sU2l6ZSwgc3RyaWRlcywgcGFkZGluZywgZGF0YUZvcm1hdCwgJ21heCcpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoTWF4UG9vbGluZzNEKTtcblxuZXhwb3J0IGNsYXNzIEF2ZXJhZ2VQb29saW5nM0QgZXh0ZW5kcyBQb29saW5nM0Qge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdBdmVyYWdlUG9vbGluZzNEJztcbiAgY29uc3RydWN0b3IoYXJnczogUG9vbGluZzNETGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgcG9vbGluZ0Z1bmN0aW9uKFxuICAgICAgaW5wdXRzOiBUZW5zb3IsIHBvb2xTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHBhZGRpbmc6IFBhZGRpbmdNb2RlLFxuICAgICAgZGF0YUZvcm1hdDogRGF0YUZvcm1hdCk6IFRlbnNvciB7XG4gICAgY2hlY2tEYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICAgIGNoZWNrUGFkZGluZ01vZGUocGFkZGluZyk7XG4gICAgcmV0dXJuIHBvb2wzZChcbiAgICAgICAgaW5wdXRzIGFzIFRlbnNvcjVELCBwb29sU2l6ZSwgc3RyaWRlcywgcGFkZGluZywgZGF0YUZvcm1hdCwgJ2F2ZycpO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQXZlcmFnZVBvb2xpbmczRCk7XG5cbi8qKlxuICogQWJzdHJhY3QgY2xhc3MgZm9yIGRpZmZlcmVudCBnbG9iYWwgcG9vbGluZyAxRCBsYXllcnMuXG4gKi9cbmV4cG9ydCBhYnN0cmFjdCBjbGFzcyBHbG9iYWxQb29saW5nMUQgZXh0ZW5kcyBMYXllciB7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDN9KV07XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGUpOiBTaGFwZSB7XG4gICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCBpbnB1dFNoYXBlWzJdXTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcigpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBHbG9iYWxBdmVyYWdlUG9vbGluZzFEIGV4dGVuZHMgR2xvYmFsUG9vbGluZzFEIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnR2xvYmFsQXZlcmFnZVBvb2xpbmcxRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzIHx8IHt9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBpbnB1dCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcbiAgICAgIHJldHVybiB0ZmMubWVhbihpbnB1dCwgMSk7XG4gICAgfSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhHbG9iYWxBdmVyYWdlUG9vbGluZzFEKTtcblxuZXhwb3J0IGNsYXNzIEdsb2JhbE1heFBvb2xpbmcxRCBleHRlbmRzIEdsb2JhbFBvb2xpbmcxRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0dsb2JhbE1heFBvb2xpbmcxRCc7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MgfHwge30pO1xuICB9XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IGlucHV0ID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgcmV0dXJuIHRmYy5tYXgoaW5wdXQsIDEpO1xuICAgIH0pO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR2xvYmFsTWF4UG9vbGluZzFEKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIEdsb2JhbFBvb2xpbmcyRExheWVyQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIC8qKlxuICAgKiBPbmUgb2YgYENIQU5ORUxfTEFTVGAgKGRlZmF1bHQpIG9yIGBDSEFOTkVMX0ZJUlNUYC5cbiAgICpcbiAgICogVGhlIG9yZGVyaW5nIG9mIHRoZSBkaW1lbnNpb25zIGluIHRoZSBpbnB1dHMuIGBDSEFOTkVMX0xBU1RgIGNvcnJlc3BvbmRzXG4gICAqIHRvIGlucHV0cyB3aXRoIHNoYXBlIGBbYmF0Y2gsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXWAgd2hpbGVcbiAgICogYENIQU5ORUxfRklSU1RgIGNvcnJlc3BvbmRzIHRvIGlucHV0cyB3aXRoIHNoYXBlXG4gICAqIGBbYmF0Y2gsIGNoYW5uZWxzLCBoZWlnaHQsIHdpZHRoXWAuXG4gICAqL1xuICBkYXRhRm9ybWF0PzogRGF0YUZvcm1hdDtcbn1cblxuLyoqXG4gKiBBYnN0cmFjdCBjbGFzcyBmb3IgZGlmZmVyZW50IGdsb2JhbCBwb29saW5nIDJEIGxheWVycy5cbiAqL1xuZXhwb3J0IGFic3RyYWN0IGNsYXNzIEdsb2JhbFBvb2xpbmcyRCBleHRlbmRzIExheWVyIHtcbiAgcHJvdGVjdGVkIGRhdGFGb3JtYXQ6IERhdGFGb3JtYXQ7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IEdsb2JhbFBvb2xpbmcyRExheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMuZGF0YUZvcm1hdCA9XG4gICAgICAgIGFyZ3MuZGF0YUZvcm1hdCA9PSBudWxsID8gJ2NoYW5uZWxzTGFzdCcgOiBhcmdzLmRhdGFGb3JtYXQ7XG4gICAgY2hlY2tEYXRhRm9ybWF0KHRoaXMuZGF0YUZvcm1hdCk7XG4gICAgdGhpcy5pbnB1dFNwZWMgPSBbbmV3IElucHV0U3BlYyh7bmRpbTogNH0pXTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaW5wdXRTaGFwZSA9IGlucHV0U2hhcGUgYXMgU2hhcGU7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVszXV07XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsxXV07XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHRocm93IG5ldyBOb3RJbXBsZW1lbnRlZEVycm9yKCk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWcgPSB7ZGF0YUZvcm1hdDogdGhpcy5kYXRhRm9ybWF0fTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEdsb2JhbEF2ZXJhZ2VQb29saW5nMkQgZXh0ZW5kcyBHbG9iYWxQb29saW5nMkQge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdHbG9iYWxBdmVyYWdlUG9vbGluZzJEJztcblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgICAgICByZXR1cm4gdGZjLm1lYW4oaW5wdXQsIFsxLCAyXSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gdGZjLm1lYW4oaW5wdXQsIFsyLCAzXSk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhHbG9iYWxBdmVyYWdlUG9vbGluZzJEKTtcblxuZXhwb3J0IGNsYXNzIEdsb2JhbE1heFBvb2xpbmcyRCBleHRlbmRzIEdsb2JhbFBvb2xpbmcyRCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0dsb2JhbE1heFBvb2xpbmcyRCc7XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpOiBUZW5zb3J8VGVuc29yW10ge1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IGlucHV0ID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICAgICAgcmV0dXJuIHRmYy5tYXgoaW5wdXQsIFsxLCAyXSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gdGZjLm1heChpbnB1dCwgWzIsIDNdKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEdsb2JhbE1heFBvb2xpbmcyRCk7XG4iXX0=