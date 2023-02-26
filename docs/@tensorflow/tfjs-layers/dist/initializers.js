/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import { eye, linalg, mul, ones, randomUniform, scalar, serialization, tidy, transpose, truncatedNormal, zeros } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { checkDataFormat } from './common';
import { NotImplementedError, ValueError } from './errors';
import { VALID_DISTRIBUTION_VALUES, VALID_FAN_MODE_VALUES } from './keras_format/initializer_config';
import { checkStringTypeUnionValue, deserializeKerasObject, serializeKerasObject } from './utils/generic_utils';
import { arrayProd } from './utils/math_utils';
export function checkFanMode(value) {
    checkStringTypeUnionValue(VALID_FAN_MODE_VALUES, 'FanMode', value);
}
export function checkDistribution(value) {
    checkStringTypeUnionValue(VALID_DISTRIBUTION_VALUES, 'Distribution', value);
}
/**
 * Initializer base class.
 *
 * @doc {
 *   heading: 'Initializers', subheading: 'Classes', namespace: 'initializers'}
 */
export class Initializer extends serialization.Serializable {
    fromConfigUsesCustomObjects() {
        return false;
    }
    getConfig() {
        return {};
    }
}
export class Zeros extends Initializer {
    apply(shape, dtype) {
        return zeros(shape, dtype);
    }
}
/** @nocollapse */
Zeros.className = 'Zeros';
serialization.registerClass(Zeros);
export class Ones extends Initializer {
    apply(shape, dtype) {
        return ones(shape, dtype);
    }
}
/** @nocollapse */
Ones.className = 'Ones';
serialization.registerClass(Ones);
export class Constant extends Initializer {
    constructor(args) {
        super();
        if (typeof args !== 'object') {
            throw new ValueError(`Expected argument of type ConstantConfig but got ${args}`);
        }
        if (args.value === undefined) {
            throw new ValueError(`config must have value set but got ${args}`);
        }
        this.value = args.value;
    }
    apply(shape, dtype) {
        return tidy(() => mul(scalar(this.value), ones(shape, dtype)));
    }
    getConfig() {
        return {
            value: this.value,
        };
    }
}
/** @nocollapse */
Constant.className = 'Constant';
serialization.registerClass(Constant);
export class RandomUniform extends Initializer {
    constructor(args) {
        super();
        this.DEFAULT_MINVAL = -0.05;
        this.DEFAULT_MAXVAL = 0.05;
        this.minval = args.minval || this.DEFAULT_MINVAL;
        this.maxval = args.maxval || this.DEFAULT_MAXVAL;
        this.seed = args.seed;
    }
    apply(shape, dtype) {
        return randomUniform(shape, this.minval, this.maxval, dtype, this.seed);
    }
    getConfig() {
        return { minval: this.minval, maxval: this.maxval, seed: this.seed };
    }
}
/** @nocollapse */
RandomUniform.className = 'RandomUniform';
serialization.registerClass(RandomUniform);
export class RandomNormal extends Initializer {
    constructor(args) {
        super();
        this.DEFAULT_MEAN = 0.;
        this.DEFAULT_STDDEV = 0.05;
        this.mean = args.mean || this.DEFAULT_MEAN;
        this.stddev = args.stddev || this.DEFAULT_STDDEV;
        this.seed = args.seed;
    }
    apply(shape, dtype) {
        dtype = dtype || 'float32';
        if (dtype !== 'float32' && dtype !== 'int32') {
            throw new NotImplementedError(`randomNormal does not support dType ${dtype}.`);
        }
        return K.randomNormal(shape, this.mean, this.stddev, dtype, this.seed);
    }
    getConfig() {
        return { mean: this.mean, stddev: this.stddev, seed: this.seed };
    }
}
/** @nocollapse */
RandomNormal.className = 'RandomNormal';
serialization.registerClass(RandomNormal);
export class TruncatedNormal extends Initializer {
    constructor(args) {
        super();
        this.DEFAULT_MEAN = 0.;
        this.DEFAULT_STDDEV = 0.05;
        this.mean = args.mean || this.DEFAULT_MEAN;
        this.stddev = args.stddev || this.DEFAULT_STDDEV;
        this.seed = args.seed;
    }
    apply(shape, dtype) {
        dtype = dtype || 'float32';
        if (dtype !== 'float32' && dtype !== 'int32') {
            throw new NotImplementedError(`truncatedNormal does not support dType ${dtype}.`);
        }
        return truncatedNormal(shape, this.mean, this.stddev, dtype, this.seed);
    }
    getConfig() {
        return { mean: this.mean, stddev: this.stddev, seed: this.seed };
    }
}
/** @nocollapse */
TruncatedNormal.className = 'TruncatedNormal';
serialization.registerClass(TruncatedNormal);
export class Identity extends Initializer {
    constructor(args) {
        super();
        this.gain = args.gain != null ? args.gain : 1.0;
    }
    apply(shape, dtype) {
        return tidy(() => {
            if (shape.length !== 2 || shape[0] !== shape[1]) {
                throw new ValueError('Identity matrix initializer can only be used for' +
                    ' 2D square matrices.');
            }
            else {
                return mul(this.gain, eye(shape[0]));
            }
        });
    }
    getConfig() {
        return { gain: this.gain };
    }
}
/** @nocollapse */
Identity.className = 'Identity';
serialization.registerClass(Identity);
/**
 * Computes the number of input and output units for a weight shape.
 * @param shape Shape of weight.
 * @param dataFormat data format to use for convolution kernels.
 *   Note that all kernels in Keras are standardized on the
 *   CHANNEL_LAST ordering (even when inputs are set to CHANNEL_FIRST).
 * @return An length-2 array: fanIn, fanOut.
 */
function computeFans(shape, dataFormat = 'channelsLast') {
    let fanIn;
    let fanOut;
    checkDataFormat(dataFormat);
    if (shape.length === 2) {
        fanIn = shape[0];
        fanOut = shape[1];
    }
    else if ([3, 4, 5].indexOf(shape.length) !== -1) {
        if (dataFormat === 'channelsFirst') {
            const receptiveFieldSize = arrayProd(shape, 2);
            fanIn = shape[1] * receptiveFieldSize;
            fanOut = shape[0] * receptiveFieldSize;
        }
        else if (dataFormat === 'channelsLast') {
            const receptiveFieldSize = arrayProd(shape, 0, shape.length - 2);
            fanIn = shape[shape.length - 2] * receptiveFieldSize;
            fanOut = shape[shape.length - 1] * receptiveFieldSize;
        }
    }
    else {
        const shapeProd = arrayProd(shape);
        fanIn = Math.sqrt(shapeProd);
        fanOut = Math.sqrt(shapeProd);
    }
    return [fanIn, fanOut];
}
export class VarianceScaling extends Initializer {
    /**
     * Constructor of VarianceScaling.
     * @throws ValueError for invalid value in scale.
     */
    constructor(args) {
        super();
        if (args.scale < 0.0) {
            throw new ValueError(`scale must be a positive float. Got: ${args.scale}`);
        }
        this.scale = args.scale == null ? 1.0 : args.scale;
        this.mode = args.mode == null ? 'fanIn' : args.mode;
        checkFanMode(this.mode);
        this.distribution =
            args.distribution == null ? 'normal' : args.distribution;
        checkDistribution(this.distribution);
        this.seed = args.seed;
    }
    apply(shape, dtype) {
        const fans = computeFans(shape);
        const fanIn = fans[0];
        const fanOut = fans[1];
        let scale = this.scale;
        if (this.mode === 'fanIn') {
            scale /= Math.max(1, fanIn);
        }
        else if (this.mode === 'fanOut') {
            scale /= Math.max(1, fanOut);
        }
        else {
            scale /= Math.max(1, (fanIn + fanOut) / 2);
        }
        if (this.distribution === 'normal') {
            const stddev = Math.sqrt(scale);
            dtype = dtype || 'float32';
            if (dtype !== 'float32' && dtype !== 'int32') {
                throw new NotImplementedError(`${this.getClassName()} does not support dType ${dtype}.`);
            }
            return truncatedNormal(shape, 0, stddev, dtype, this.seed);
        }
        else {
            const limit = Math.sqrt(3 * scale);
            return randomUniform(shape, -limit, limit, dtype, this.seed);
        }
    }
    getConfig() {
        return {
            scale: this.scale,
            mode: this.mode,
            distribution: this.distribution,
            seed: this.seed
        };
    }
}
/** @nocollapse */
VarianceScaling.className = 'VarianceScaling';
serialization.registerClass(VarianceScaling);
export class GlorotUniform extends VarianceScaling {
    /**
     * Constructor of GlorotUniform
     * @param scale
     * @param mode
     * @param distribution
     * @param seed
     */
    constructor(args) {
        super({
            scale: 1.0,
            mode: 'fanAvg',
            distribution: 'uniform',
            seed: args == null ? null : args.seed
        });
    }
    getClassName() {
        // In Python Keras, GlorotUniform is not a class, but a helper method
        // that creates a VarianceScaling object. Use 'VarianceScaling' as
        // class name to be compatible with that.
        return VarianceScaling.className;
    }
}
/** @nocollapse */
GlorotUniform.className = 'GlorotUniform';
serialization.registerClass(GlorotUniform);
export class GlorotNormal extends VarianceScaling {
    /**
     * Constructor of GlorotNormal.
     * @param scale
     * @param mode
     * @param distribution
     * @param seed
     */
    constructor(args) {
        super({
            scale: 1.0,
            mode: 'fanAvg',
            distribution: 'normal',
            seed: args == null ? null : args.seed
        });
    }
    getClassName() {
        // In Python Keras, GlorotNormal is not a class, but a helper method
        // that creates a VarianceScaling object. Use 'VarianceScaling' as
        // class name to be compatible with that.
        return VarianceScaling.className;
    }
}
/** @nocollapse */
GlorotNormal.className = 'GlorotNormal';
serialization.registerClass(GlorotNormal);
export class HeNormal extends VarianceScaling {
    constructor(args) {
        super({
            scale: 2.0,
            mode: 'fanIn',
            distribution: 'normal',
            seed: args == null ? null : args.seed
        });
    }
    getClassName() {
        // In Python Keras, HeNormal is not a class, but a helper method
        // that creates a VarianceScaling object. Use 'VarianceScaling' as
        // class name to be compatible with that.
        return VarianceScaling.className;
    }
}
/** @nocollapse */
HeNormal.className = 'HeNormal';
serialization.registerClass(HeNormal);
export class HeUniform extends VarianceScaling {
    constructor(args) {
        super({
            scale: 2.0,
            mode: 'fanIn',
            distribution: 'uniform',
            seed: args == null ? null : args.seed
        });
    }
    getClassName() {
        // In Python Keras, HeUniform is not a class, but a helper method
        // that creates a VarianceScaling object. Use 'VarianceScaling' as
        // class name to be compatible with that.
        return VarianceScaling.className;
    }
}
/** @nocollapse */
HeUniform.className = 'HeUniform';
serialization.registerClass(HeUniform);
export class LeCunNormal extends VarianceScaling {
    constructor(args) {
        super({
            scale: 1.0,
            mode: 'fanIn',
            distribution: 'normal',
            seed: args == null ? null : args.seed
        });
    }
    getClassName() {
        // In Python Keras, LeCunNormal is not a class, but a helper method
        // that creates a VarianceScaling object. Use 'VarianceScaling' as
        // class name to be compatible with that.
        return VarianceScaling.className;
    }
}
/** @nocollapse */
LeCunNormal.className = 'LeCunNormal';
serialization.registerClass(LeCunNormal);
export class LeCunUniform extends VarianceScaling {
    constructor(args) {
        super({
            scale: 1.0,
            mode: 'fanIn',
            distribution: 'uniform',
            seed: args == null ? null : args.seed
        });
    }
    getClassName() {
        // In Python Keras, LeCunUniform is not a class, but a helper method
        // that creates a VarianceScaling object. Use 'VarianceScaling' as
        // class name to be compatible with that.
        return VarianceScaling.className;
    }
}
/** @nocollapse */
LeCunUniform.className = 'LeCunUniform';
serialization.registerClass(LeCunUniform);
export class Orthogonal extends Initializer {
    constructor(args) {
        super();
        this.DEFAULT_GAIN = 1;
        this.gain = args.gain == null ? this.DEFAULT_GAIN : args.gain;
        this.seed = args.seed;
        if (this.seed != null) {
            throw new NotImplementedError('Random seed is not implemented for Orthogonal Initializer yet.');
        }
    }
    apply(shape, dtype) {
        return tidy(() => {
            if (shape.length < 2) {
                throw new NotImplementedError('Shape must be at least 2D.');
            }
            if (shape[0] * shape[1] > 2000) {
                console.warn(`Orthogonal initializer is being called on a matrix with more ` +
                    `than 2000 (${shape[0] * shape[1]}) elements: ` +
                    `Slowness may result.`);
            }
            // TODO(cais): Add seed support.
            const normalizedShape = shape[0] > shape[1] ? [shape[1], shape[0]] : shape;
            const a = K.randomNormal(normalizedShape, 0, 1, 'float32');
            let q = linalg.gramSchmidt(a);
            if (shape[0] > shape[1]) {
                q = transpose(q);
            }
            return mul(this.gain, q);
        });
    }
    getConfig() {
        return {
            gain: this.gain,
            seed: this.seed,
        };
    }
}
/** @nocollapse */
Orthogonal.className = 'Orthogonal';
serialization.registerClass(Orthogonal);
// Maps the JavaScript-like identifier keys to the corresponding registry
// symbols.
export const INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP = {
    'constant': 'Constant',
    'glorotNormal': 'GlorotNormal',
    'glorotUniform': 'GlorotUniform',
    'heNormal': 'HeNormal',
    'heUniform': 'HeUniform',
    'identity': 'Identity',
    'leCunNormal': 'LeCunNormal',
    'leCunUniform': 'LeCunUniform',
    'ones': 'Ones',
    'orthogonal': 'Orthogonal',
    'randomNormal': 'RandomNormal',
    'randomUniform': 'RandomUniform',
    'truncatedNormal': 'TruncatedNormal',
    'varianceScaling': 'VarianceScaling',
    'zeros': 'Zeros'
};
function deserializeInitializer(config, customObjects = {}) {
    return deserializeKerasObject(config, serialization.SerializationMap.getMap().classNameMap, customObjects, 'initializer');
}
export function serializeInitializer(initializer) {
    return serializeKerasObject(initializer);
}
export function getInitializer(identifier) {
    if (typeof identifier === 'string') {
        const className = identifier in INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
            INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
            identifier;
        /* We have four 'helper' classes for common initializers that
        all get serialized as 'VarianceScaling' and shouldn't go through
        the deserializeInitializer pathway. */
        if (className === 'GlorotNormal') {
            return new GlorotNormal();
        }
        else if (className === 'GlorotUniform') {
            return new GlorotUniform();
        }
        else if (className === 'HeNormal') {
            return new HeNormal();
        }
        else if (className === 'HeUniform') {
            return new HeUniform();
        }
        else if (className === 'LeCunNormal') {
            return new LeCunNormal();
        }
        else if (className === 'LeCunUniform') {
            return new LeCunUniform();
        }
        else {
            const config = {};
            config['className'] = className;
            config['config'] = {};
            return deserializeInitializer(config);
        }
    }
    else if (identifier instanceof Initializer) {
        return identifier;
    }
    else {
        return deserializeInitializer(identifier);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5pdGlhbGl6ZXJzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2luaXRpYWxpemVycy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVILE9BQU8sRUFBVyxHQUFHLEVBQUUsTUFBTSxFQUFFLEdBQUcsRUFBRSxJQUFJLEVBQUUsYUFBYSxFQUFFLE1BQU0sRUFBRSxhQUFhLEVBQW9CLElBQUksRUFBRSxTQUFTLEVBQUUsZUFBZSxFQUFFLEtBQUssRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXhLLE9BQU8sS0FBSyxDQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDNUMsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUN6QyxPQUFPLEVBQUMsbUJBQW1CLEVBQUUsVUFBVSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBRXpELE9BQU8sRUFBd0IseUJBQXlCLEVBQUUscUJBQXFCLEVBQUMsTUFBTSxtQ0FBbUMsQ0FBQztBQUMxSCxPQUFPLEVBQUMseUJBQXlCLEVBQUUsc0JBQXNCLEVBQUUsb0JBQW9CLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUM5RyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFN0MsTUFBTSxVQUFVLFlBQVksQ0FBQyxLQUFjO0lBQ3pDLHlCQUF5QixDQUFDLHFCQUFxQixFQUFFLFNBQVMsRUFBRSxLQUFLLENBQUMsQ0FBQztBQUNyRSxDQUFDO0FBRUQsTUFBTSxVQUFVLGlCQUFpQixDQUFDLEtBQWM7SUFDOUMseUJBQXlCLENBQUMseUJBQXlCLEVBQUUsY0FBYyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQzlFLENBQUM7QUFFRDs7Ozs7R0FLRztBQUNILE1BQU0sT0FBZ0IsV0FBWSxTQUFRLGFBQWEsQ0FBQyxZQUFZO0lBQzNELDJCQUEyQjtRQUNoQyxPQUFPLEtBQUssQ0FBQztJQUNmLENBQUM7SUFTRCxT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MEMsQ0FBQyxDQUFDO1lBQ3hFLDBDQUEwQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUM7WUFDeEQsVUFBVSxDQUFDO1FBQ2Y7OzhDQUVzQztRQUN0QyxJQUFJLFNBQVMsS0FBSyxjQUFjLEVBQUU7WUFDaEMsT0FBTyxJQUFJLFlBQVksRUFBRSxDQUFDO1NBQzNCO2FBQU0sSUFBSSxTQUFTLEtBQUssZUFBZSxFQUFFO1lBQ3hDLE9BQU8sSUFBSSxhQUFhLEVBQUUsQ0FBQztTQUM1QjthQUFNLElBQUksU0FBUyxLQUFLLFVBQVUsRUFBRTtZQUNuQyxPQUFPLElBQUksUUFBUSxFQUFFLENBQUM7U0FDdkI7YUFBTSxJQUFJLFNBQVMsS0FBSyxXQUFXLEVBQUU7WUFDcEMsT0FBTyxJQUFJLFNBQVMsRUFBRSxDQUFDO1NBQ3hCO2FBQU0sSUFBSSxTQUFTLEtBQUssYUFBYSxFQUFFO1lBQ3RDLE9BQU8sSUFBSSxXQUFXLEVBQUUsQ0FBQztTQUMxQjthQUFNLElBQUksU0FBUyxLQUFLLGNBQWMsRUFBRTtZQUN2QyxPQUFPLElBQUksWUFBWSxFQUFFLENBQUM7U0FDM0I7YUFBTTtZQUNMLE1BQU0sTUFBTSxHQUE2QixFQUFFLENBQUM7WUFDNUMsTUFBTSxDQUFDLFdBQVcsQ0FBQyxHQUFHLFNBQVMsQ0FBQztZQUNoQyxNQUFNLENBQUMsUUFBUSxDQUFDLEdBQUcsRUFBRSxDQUFDO1lBQ3RCLE9BQU8sc0JBQXNCLENBQUMsTUFBTSxDQUFDLENBQUM7U0FDdkM7S0FDRjtTQUFNLElBQUksVUFBVSxZQUFZLFdBQVcsRUFBRTtRQUM1QyxPQUFPLFVBQVUsQ0FBQztLQUNuQjtTQUFNO1FBQ0wsT0FBTyxzQkFBc0IsQ0FBQyxVQUFVLENBQUMsQ0FBQztLQUMzQztBQUNILENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0RhdGFUeXBlLCBleWUsIGxpbmFsZywgbXVsLCBvbmVzLCByYW5kb21Vbmlmb3JtLCBzY2FsYXIsIHNlcmlhbGl6YXRpb24sIFRlbnNvciwgVGVuc29yMkQsIHRpZHksIHRyYW5zcG9zZSwgdHJ1bmNhdGVkTm9ybWFsLCB6ZXJvc30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0ICogYXMgSyBmcm9tICcuL2JhY2tlbmQvdGZqc19iYWNrZW5kJztcbmltcG9ydCB7Y2hlY2tEYXRhRm9ybWF0fSBmcm9tICcuL2NvbW1vbic7XG5pbXBvcnQge05vdEltcGxlbWVudGVkRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4vZXJyb3JzJztcbmltcG9ydCB7RGF0YUZvcm1hdCwgU2hhcGV9IGZyb20gJy4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge0Rpc3RyaWJ1dGlvbiwgRmFuTW9kZSwgVkFMSURfRElTVFJJQlVUSU9OX1ZBTFVFUywgVkFMSURfRkFOX01PREVfVkFMVUVTfSBmcm9tICcuL2tlcmFzX2Zvcm1hdC9pbml0aWFsaXplcl9jb25maWcnO1xuaW1wb3J0IHtjaGVja1N0cmluZ1R5cGVVbmlvblZhbHVlLCBkZXNlcmlhbGl6ZUtlcmFzT2JqZWN0LCBzZXJpYWxpemVLZXJhc09iamVjdH0gZnJvbSAnLi91dGlscy9nZW5lcmljX3V0aWxzJztcbmltcG9ydCB7YXJyYXlQcm9kfSBmcm9tICcuL3V0aWxzL21hdGhfdXRpbHMnO1xuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tGYW5Nb2RlKHZhbHVlPzogc3RyaW5nKTogdm9pZCB7XG4gIGNoZWNrU3RyaW5nVHlwZVVuaW9uVmFsdWUoVkFMSURfRkFOX01PREVfVkFMVUVTLCAnRmFuTW9kZScsIHZhbHVlKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNoZWNrRGlzdHJpYnV0aW9uKHZhbHVlPzogc3RyaW5nKTogdm9pZCB7XG4gIGNoZWNrU3RyaW5nVHlwZVVuaW9uVmFsdWUoVkFMSURfRElTVFJJQlVUSU9OX1ZBTFVFUywgJ0Rpc3RyaWJ1dGlvbicsIHZhbHVlKTtcbn1cblxuLyoqXG4gKiBJbml0aWFsaXplciBiYXNlIGNsYXNzLlxuICpcbiAqIEBkb2Mge1xuICogICBoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnLCBuYW1lc3BhY2U6ICdpbml0aWFsaXplcnMnfVxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgSW5pdGlhbGl6ZXIgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZSB7XG4gIHB1YmxpYyBmcm9tQ29uZmlnVXNlc0N1c3RvbU9iamVjdHMoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIC8qKlxuICAgKiBHZW5lcmF0ZSBhbiBpbml0aWFsIHZhbHVlLlxuICAgKiBAcGFyYW0gc2hhcGVcbiAgICogQHBhcmFtIGR0eXBlXG4gICAqIEByZXR1cm4gVGhlIGluaXQgdmFsdWUuXG4gICAqL1xuICBhYnN0cmFjdCBhcHBseShzaGFwZTogU2hhcGUsIGR0eXBlPzogRGF0YVR5cGUpOiBUZW5zb3I7XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgcmV0dXJuIHt9O1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBaZXJvcyBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnWmVyb3MnO1xuXG4gIGFwcGx5KHNoYXBlOiBTaGFwZSwgZHR5cGU/OiBEYXRhVHlwZSk6IFRlbnNvciB7XG4gICAgcmV0dXJuIHplcm9zKHNoYXBlLCBkdHlwZSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhaZXJvcyk7XG5cbmV4cG9ydCBjbGFzcyBPbmVzIGV4dGVuZHMgSW5pdGlhbGl6ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdPbmVzJztcblxuICBhcHBseShzaGFwZTogU2hhcGUsIGR0eXBlPzogRGF0YVR5cGUpOiBUZW5zb3Ige1xuICAgIHJldHVybiBvbmVzKHNoYXBlLCBkdHlwZSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhPbmVzKTtcblxuZXhwb3J0IGludGVyZmFjZSBDb25zdGFudEFyZ3Mge1xuICAvKiogVGhlIHZhbHVlIGZvciBlYWNoIGVsZW1lbnQgaW4gdGhlIHZhcmlhYmxlLiAqL1xuICB2YWx1ZTogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgQ29uc3RhbnQgZXh0ZW5kcyBJbml0aWFsaXplciB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0NvbnN0YW50JztcbiAgcHJpdmF0ZSB2YWx1ZTogbnVtYmVyO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBDb25zdGFudEFyZ3MpIHtcbiAgICBzdXBlcigpO1xuICAgIGlmICh0eXBlb2YgYXJncyAhPT0gJ29iamVjdCcpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBFeHBlY3RlZCBhcmd1bWVudCBvZiB0eXBlIENvbnN0YW50Q29uZmlnIGJ1dCBnb3QgJHthcmdzfWApO1xuICAgIH1cbiAgICBpZiAoYXJncy52YWx1ZSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihgY29uZmlnIG11c3QgaGF2ZSB2YWx1ZSBzZXQgYnV0IGdvdCAke2FyZ3N9YCk7XG4gICAgfVxuICAgIHRoaXMudmFsdWUgPSBhcmdzLnZhbHVlO1xuICB9XG5cbiAgYXBwbHkoc2hhcGU6IFNoYXBlLCBkdHlwZT86IERhdGFUeXBlKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiBtdWwoc2NhbGFyKHRoaXMudmFsdWUpLCBvbmVzKHNoYXBlLCBkdHlwZSkpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7XG4gICAgICB2YWx1ZTogdGhpcy52YWx1ZSxcbiAgICB9O1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoQ29uc3RhbnQpO1xuXG5leHBvcnQgaW50ZXJmYWNlIFJhbmRvbVVuaWZvcm1BcmdzIHtcbiAgLyoqIExvd2VyIGJvdW5kIG9mIHRoZSByYW5nZSBvZiByYW5kb20gdmFsdWVzIHRvIGdlbmVyYXRlLiAqL1xuICBtaW52YWw/OiBudW1iZXI7XG4gIC8qKiBVcHBlciBib3VuZCBvZiB0aGUgcmFuZ2Ugb2YgcmFuZG9tIHZhbHVlcyB0byBnZW5lcmF0ZS4gKi9cbiAgbWF4dmFsPzogbnVtYmVyO1xuICAvKiogVXNlZCB0byBzZWVkIHRoZSByYW5kb20gZ2VuZXJhdG9yLiAqL1xuICBzZWVkPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgUmFuZG9tVW5pZm9ybSBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnUmFuZG9tVW5pZm9ybSc7XG4gIHJlYWRvbmx5IERFRkFVTFRfTUlOVkFMID0gLTAuMDU7XG4gIHJlYWRvbmx5IERFRkFVTFRfTUFYVkFMID0gMC4wNTtcbiAgcHJpdmF0ZSBtaW52YWw6IG51bWJlcjtcbiAgcHJpdmF0ZSBtYXh2YWw6IG51bWJlcjtcbiAgcHJpdmF0ZSBzZWVkOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoYXJnczogUmFuZG9tVW5pZm9ybUFyZ3MpIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMubWludmFsID0gYXJncy5taW52YWwgfHwgdGhpcy5ERUZBVUxUX01JTlZBTDtcbiAgICB0aGlzLm1heHZhbCA9IGFyZ3MubWF4dmFsIHx8IHRoaXMuREVGQVVMVF9NQVhWQUw7XG4gICAgdGhpcy5zZWVkID0gYXJncy5zZWVkO1xuICB9XG5cbiAgYXBwbHkoc2hhcGU6IFNoYXBlLCBkdHlwZT86IERhdGFUeXBlKTogVGVuc29yIHtcbiAgICByZXR1cm4gcmFuZG9tVW5pZm9ybShzaGFwZSwgdGhpcy5taW52YWwsIHRoaXMubWF4dmFsLCBkdHlwZSwgdGhpcy5zZWVkKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7bWludmFsOiB0aGlzLm1pbnZhbCwgbWF4dmFsOiB0aGlzLm1heHZhbCwgc2VlZDogdGhpcy5zZWVkfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFJhbmRvbVVuaWZvcm0pO1xuXG5leHBvcnQgaW50ZXJmYWNlIFJhbmRvbU5vcm1hbEFyZ3Mge1xuICAvKiogTWVhbiBvZiB0aGUgcmFuZG9tIHZhbHVlcyB0byBnZW5lcmF0ZS4gKi9cbiAgbWVhbj86IG51bWJlcjtcbiAgLyoqIFN0YW5kYXJkIGRldmlhdGlvbiBvZiB0aGUgcmFuZG9tIHZhbHVlcyB0byBnZW5lcmF0ZS4gKi9cbiAgc3RkZGV2PzogbnVtYmVyO1xuICAvKiogVXNlZCB0byBzZWVkIHRoZSByYW5kb20gZ2VuZXJhdG9yLiAqL1xuICBzZWVkPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgUmFuZG9tTm9ybWFsIGV4dGVuZHMgSW5pdGlhbGl6ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdSYW5kb21Ob3JtYWwnO1xuICByZWFkb25seSBERUZBVUxUX01FQU4gPSAwLjtcbiAgcmVhZG9ubHkgREVGQVVMVF9TVERERVYgPSAwLjA1O1xuICBwcml2YXRlIG1lYW46IG51bWJlcjtcbiAgcHJpdmF0ZSBzdGRkZXY6IG51bWJlcjtcbiAgcHJpdmF0ZSBzZWVkOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoYXJnczogUmFuZG9tTm9ybWFsQXJncykge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5tZWFuID0gYXJncy5tZWFuIHx8IHRoaXMuREVGQVVMVF9NRUFOO1xuICAgIHRoaXMuc3RkZGV2ID0gYXJncy5zdGRkZXYgfHwgdGhpcy5ERUZBVUxUX1NURERFVjtcbiAgICB0aGlzLnNlZWQgPSBhcmdzLnNlZWQ7XG4gIH1cblxuICBhcHBseShzaGFwZTogU2hhcGUsIGR0eXBlPzogRGF0YVR5cGUpOiBUZW5zb3Ige1xuICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIGlmIChkdHlwZSAhPT0gJ2Zsb2F0MzInICYmIGR0eXBlICE9PSAnaW50MzInKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICBgcmFuZG9tTm9ybWFsIGRvZXMgbm90IHN1cHBvcnQgZFR5cGUgJHtkdHlwZX0uYCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIEsucmFuZG9tTm9ybWFsKHNoYXBlLCB0aGlzLm1lYW4sIHRoaXMuc3RkZGV2LCBkdHlwZSwgdGhpcy5zZWVkKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7bWVhbjogdGhpcy5tZWFuLCBzdGRkZXY6IHRoaXMuc3RkZGV2LCBzZWVkOiB0aGlzLnNlZWR9O1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoUmFuZG9tTm9ybWFsKTtcblxuZXhwb3J0IGludGVyZmFjZSBUcnVuY2F0ZWROb3JtYWxBcmdzIHtcbiAgLyoqIE1lYW4gb2YgdGhlIHJhbmRvbSB2YWx1ZXMgdG8gZ2VuZXJhdGUuICovXG4gIG1lYW4/OiBudW1iZXI7XG4gIC8qKiBTdGFuZGFyZCBkZXZpYXRpb24gb2YgdGhlIHJhbmRvbSB2YWx1ZXMgdG8gZ2VuZXJhdGUuICovXG4gIHN0ZGRldj86IG51bWJlcjtcbiAgLyoqIFVzZWQgdG8gc2VlZCB0aGUgcmFuZG9tIGdlbmVyYXRvci4gKi9cbiAgc2VlZD86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIFRydW5jYXRlZE5vcm1hbCBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnVHJ1bmNhdGVkTm9ybWFsJztcblxuICByZWFkb25seSBERUZBVUxUX01FQU4gPSAwLjtcbiAgcmVhZG9ubHkgREVGQVVMVF9TVERERVYgPSAwLjA1O1xuICBwcml2YXRlIG1lYW46IG51bWJlcjtcbiAgcHJpdmF0ZSBzdGRkZXY6IG51bWJlcjtcbiAgcHJpdmF0ZSBzZWVkOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoYXJnczogVHJ1bmNhdGVkTm9ybWFsQXJncykge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5tZWFuID0gYXJncy5tZWFuIHx8IHRoaXMuREVGQVVMVF9NRUFOO1xuICAgIHRoaXMuc3RkZGV2ID0gYXJncy5zdGRkZXYgfHwgdGhpcy5ERUZBVUxUX1NURERFVjtcbiAgICB0aGlzLnNlZWQgPSBhcmdzLnNlZWQ7XG4gIH1cblxuICBhcHBseShzaGFwZTogU2hhcGUsIGR0eXBlPzogRGF0YVR5cGUpOiBUZW5zb3Ige1xuICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIGlmIChkdHlwZSAhPT0gJ2Zsb2F0MzInICYmIGR0eXBlICE9PSAnaW50MzInKSB7XG4gICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcihcbiAgICAgICAgICBgdHJ1bmNhdGVkTm9ybWFsIGRvZXMgbm90IHN1cHBvcnQgZFR5cGUgJHtkdHlwZX0uYCk7XG4gICAgfVxuICAgIHJldHVybiB0cnVuY2F0ZWROb3JtYWwoc2hhcGUsIHRoaXMubWVhbiwgdGhpcy5zdGRkZXYsIGR0eXBlLCB0aGlzLnNlZWQpO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgcmV0dXJuIHttZWFuOiB0aGlzLm1lYW4sIHN0ZGRldjogdGhpcy5zdGRkZXYsIHNlZWQ6IHRoaXMuc2VlZH07XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhUcnVuY2F0ZWROb3JtYWwpO1xuXG5leHBvcnQgaW50ZXJmYWNlIElkZW50aXR5QXJncyB7XG4gIC8qKlxuICAgKiBNdWx0aXBsaWNhdGl2ZSBmYWN0b3IgdG8gYXBwbHkgdG8gdGhlIGlkZW50aXR5IG1hdHJpeC5cbiAgICovXG4gIGdhaW4/OiBudW1iZXI7XG59XG5cbmV4cG9ydCBjbGFzcyBJZGVudGl0eSBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnSWRlbnRpdHknO1xuICBwcml2YXRlIGdhaW46IG51bWJlcjtcbiAgY29uc3RydWN0b3IoYXJnczogSWRlbnRpdHlBcmdzKSB7XG4gICAgc3VwZXIoKTtcbiAgICB0aGlzLmdhaW4gPSBhcmdzLmdhaW4gIT0gbnVsbCA/IGFyZ3MuZ2FpbiA6IDEuMDtcbiAgfVxuXG4gIGFwcGx5KHNoYXBlOiBTaGFwZSwgZHR5cGU/OiBEYXRhVHlwZSk6IFRlbnNvciB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgaWYgKHNoYXBlLmxlbmd0aCAhPT0gMiB8fCBzaGFwZVswXSAhPT0gc2hhcGVbMV0pIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAnSWRlbnRpdHkgbWF0cml4IGluaXRpYWxpemVyIGNhbiBvbmx5IGJlIHVzZWQgZm9yJyArXG4gICAgICAgICAgICAnIDJEIHNxdWFyZSBtYXRyaWNlcy4nKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBtdWwodGhpcy5nYWluLCBleWUoc2hhcGVbMF0pKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7Z2FpbjogdGhpcy5nYWlufTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKElkZW50aXR5KTtcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgbnVtYmVyIG9mIGlucHV0IGFuZCBvdXRwdXQgdW5pdHMgZm9yIGEgd2VpZ2h0IHNoYXBlLlxuICogQHBhcmFtIHNoYXBlIFNoYXBlIG9mIHdlaWdodC5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0IGRhdGEgZm9ybWF0IHRvIHVzZSBmb3IgY29udm9sdXRpb24ga2VybmVscy5cbiAqICAgTm90ZSB0aGF0IGFsbCBrZXJuZWxzIGluIEtlcmFzIGFyZSBzdGFuZGFyZGl6ZWQgb24gdGhlXG4gKiAgIENIQU5ORUxfTEFTVCBvcmRlcmluZyAoZXZlbiB3aGVuIGlucHV0cyBhcmUgc2V0IHRvIENIQU5ORUxfRklSU1QpLlxuICogQHJldHVybiBBbiBsZW5ndGgtMiBhcnJheTogZmFuSW4sIGZhbk91dC5cbiAqL1xuZnVuY3Rpb24gY29tcHV0ZUZhbnMoXG4gICAgc2hhcGU6IFNoYXBlLCBkYXRhRm9ybWF0OiBEYXRhRm9ybWF0ID0gJ2NoYW5uZWxzTGFzdCcpOiBudW1iZXJbXSB7XG4gIGxldCBmYW5JbjogbnVtYmVyO1xuICBsZXQgZmFuT3V0OiBudW1iZXI7XG4gIGNoZWNrRGF0YUZvcm1hdChkYXRhRm9ybWF0KTtcbiAgaWYgKHNoYXBlLmxlbmd0aCA9PT0gMikge1xuICAgIGZhbkluID0gc2hhcGVbMF07XG4gICAgZmFuT3V0ID0gc2hhcGVbMV07XG4gIH0gZWxzZSBpZiAoWzMsIDQsIDVdLmluZGV4T2Yoc2hhcGUubGVuZ3RoKSAhPT0gLTEpIHtcbiAgICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICBjb25zdCByZWNlcHRpdmVGaWVsZFNpemUgPSBhcnJheVByb2Qoc2hhcGUsIDIpO1xuICAgICAgZmFuSW4gPSBzaGFwZVsxXSAqIHJlY2VwdGl2ZUZpZWxkU2l6ZTtcbiAgICAgIGZhbk91dCA9IHNoYXBlWzBdICogcmVjZXB0aXZlRmllbGRTaXplO1xuICAgIH0gZWxzZSBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICAgIGNvbnN0IHJlY2VwdGl2ZUZpZWxkU2l6ZSA9IGFycmF5UHJvZChzaGFwZSwgMCwgc2hhcGUubGVuZ3RoIC0gMik7XG4gICAgICBmYW5JbiA9IHNoYXBlW3NoYXBlLmxlbmd0aCAtIDJdICogcmVjZXB0aXZlRmllbGRTaXplO1xuICAgICAgZmFuT3V0ID0gc2hhcGVbc2hhcGUubGVuZ3RoIC0gMV0gKiByZWNlcHRpdmVGaWVsZFNpemU7XG4gICAgfVxuICB9IGVsc2Uge1xuICAgIGNvbnN0IHNoYXBlUHJvZCA9IGFycmF5UHJvZChzaGFwZSk7XG4gICAgZmFuSW4gPSBNYXRoLnNxcnQoc2hhcGVQcm9kKTtcbiAgICBmYW5PdXQgPSBNYXRoLnNxcnQoc2hhcGVQcm9kKTtcbiAgfVxuXG4gIHJldHVybiBbZmFuSW4sIGZhbk91dF07XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgVmFyaWFuY2VTY2FsaW5nQXJncyB7XG4gIC8qKiBTY2FsaW5nIGZhY3RvciAocG9zaXRpdmUgZmxvYXQpLiAqL1xuICBzY2FsZT86IG51bWJlcjtcblxuICAvKiogRmFubmluZyBtb2RlIGZvciBpbnB1dHMgYW5kIG91dHB1dHMuICovXG4gIG1vZGU/OiBGYW5Nb2RlO1xuXG4gIC8qKiBQcm9iYWJpbGlzdGljIGRpc3RyaWJ1dGlvbiBvZiB0aGUgdmFsdWVzLiAqL1xuICBkaXN0cmlidXRpb24/OiBEaXN0cmlidXRpb247XG5cbiAgLyoqIFJhbmRvbSBudW1iZXIgZ2VuZXJhdG9yIHNlZWQuICovXG4gIHNlZWQ/OiBudW1iZXI7XG59XG5cbmV4cG9ydCBjbGFzcyBWYXJpYW5jZVNjYWxpbmcgZXh0ZW5kcyBJbml0aWFsaXplciB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ1ZhcmlhbmNlU2NhbGluZyc7XG4gIHByaXZhdGUgc2NhbGU6IG51bWJlcjtcbiAgcHJpdmF0ZSBtb2RlOiBGYW5Nb2RlO1xuICBwcml2YXRlIGRpc3RyaWJ1dGlvbjogRGlzdHJpYnV0aW9uO1xuICBwcml2YXRlIHNlZWQ6IG51bWJlcjtcblxuICAvKipcbiAgICogQ29uc3RydWN0b3Igb2YgVmFyaWFuY2VTY2FsaW5nLlxuICAgKiBAdGhyb3dzIFZhbHVlRXJyb3IgZm9yIGludmFsaWQgdmFsdWUgaW4gc2NhbGUuXG4gICAqL1xuICBjb25zdHJ1Y3RvcihhcmdzOiBWYXJpYW5jZVNjYWxpbmdBcmdzKSB7XG4gICAgc3VwZXIoKTtcbiAgICBpZiAoYXJncy5zY2FsZSA8IDAuMCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYHNjYWxlIG11c3QgYmUgYSBwb3NpdGl2ZSBmbG9hdC4gR290OiAke2FyZ3Muc2NhbGV9YCk7XG4gICAgfVxuICAgIHRoaXMuc2NhbGUgPSBhcmdzLnNjYWxlID09IG51bGwgPyAxLjAgOiBhcmdzLnNjYWxlO1xuICAgIHRoaXMubW9kZSA9IGFyZ3MubW9kZSA9PSBudWxsID8gJ2ZhbkluJyA6IGFyZ3MubW9kZTtcbiAgICBjaGVja0Zhbk1vZGUodGhpcy5tb2RlKTtcbiAgICB0aGlzLmRpc3RyaWJ1dGlvbiA9XG4gICAgICAgIGFyZ3MuZGlzdHJpYnV0aW9uID09IG51bGwgPyAnbm9ybWFsJyA6IGFyZ3MuZGlzdHJpYnV0aW9uO1xuICAgIGNoZWNrRGlzdHJpYnV0aW9uKHRoaXMuZGlzdHJpYnV0aW9uKTtcbiAgICB0aGlzLnNlZWQgPSBhcmdzLnNlZWQ7XG4gIH1cblxuICBhcHBseShzaGFwZTogU2hhcGUsIGR0eXBlPzogRGF0YVR5cGUpOiBUZW5zb3Ige1xuICAgIGNvbnN0IGZhbnMgPSBjb21wdXRlRmFucyhzaGFwZSk7XG4gICAgY29uc3QgZmFuSW4gPSBmYW5zWzBdO1xuICAgIGNvbnN0IGZhbk91dCA9IGZhbnNbMV07XG4gICAgbGV0IHNjYWxlID0gdGhpcy5zY2FsZTtcbiAgICBpZiAodGhpcy5tb2RlID09PSAnZmFuSW4nKSB7XG4gICAgICBzY2FsZSAvPSBNYXRoLm1heCgxLCBmYW5Jbik7XG4gICAgfSBlbHNlIGlmICh0aGlzLm1vZGUgPT09ICdmYW5PdXQnKSB7XG4gICAgICBzY2FsZSAvPSBNYXRoLm1heCgxLCBmYW5PdXQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBzY2FsZSAvPSBNYXRoLm1heCgxLCAoZmFuSW4gKyBmYW5PdXQpIC8gMik7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMuZGlzdHJpYnV0aW9uID09PSAnbm9ybWFsJykge1xuICAgICAgY29uc3Qgc3RkZGV2ID0gTWF0aC5zcXJ0KHNjYWxlKTtcbiAgICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgICAgaWYgKGR0eXBlICE9PSAnZmxvYXQzMicgJiYgZHR5cGUgIT09ICdpbnQzMicpIHtcbiAgICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgICBgJHt0aGlzLmdldENsYXNzTmFtZSgpfSBkb2VzIG5vdCBzdXBwb3J0IGRUeXBlICR7ZHR5cGV9LmApO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHRydW5jYXRlZE5vcm1hbChzaGFwZSwgMCwgc3RkZGV2LCBkdHlwZSwgdGhpcy5zZWVkKTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgbGltaXQgPSBNYXRoLnNxcnQoMyAqIHNjYWxlKTtcbiAgICAgIHJldHVybiByYW5kb21Vbmlmb3JtKHNoYXBlLCAtbGltaXQsIGxpbWl0LCBkdHlwZSwgdGhpcy5zZWVkKTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICByZXR1cm4ge1xuICAgICAgc2NhbGU6IHRoaXMuc2NhbGUsXG4gICAgICBtb2RlOiB0aGlzLm1vZGUsXG4gICAgICBkaXN0cmlidXRpb246IHRoaXMuZGlzdHJpYnV0aW9uLFxuICAgICAgc2VlZDogdGhpcy5zZWVkXG4gICAgfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFZhcmlhbmNlU2NhbGluZyk7XG5cbmV4cG9ydCBpbnRlcmZhY2UgU2VlZE9ubHlJbml0aWFsaXplckFyZ3Mge1xuICAvKiogUmFuZG9tIG51bWJlciBnZW5lcmF0b3Igc2VlZC4gKi9cbiAgc2VlZD86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIEdsb3JvdFVuaWZvcm0gZXh0ZW5kcyBWYXJpYW5jZVNjYWxpbmcge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGNsYXNzTmFtZSA9ICdHbG9yb3RVbmlmb3JtJztcblxuICAvKipcbiAgICogQ29uc3RydWN0b3Igb2YgR2xvcm90VW5pZm9ybVxuICAgKiBAcGFyYW0gc2NhbGVcbiAgICogQHBhcmFtIG1vZGVcbiAgICogQHBhcmFtIGRpc3RyaWJ1dGlvblxuICAgKiBAcGFyYW0gc2VlZFxuICAgKi9cbiAgY29uc3RydWN0b3IoYXJncz86IFNlZWRPbmx5SW5pdGlhbGl6ZXJBcmdzKSB7XG4gICAgc3VwZXIoe1xuICAgICAgc2NhbGU6IDEuMCxcbiAgICAgIG1vZGU6ICdmYW5BdmcnLFxuICAgICAgZGlzdHJpYnV0aW9uOiAndW5pZm9ybScsXG4gICAgICBzZWVkOiBhcmdzID09IG51bGwgPyBudWxsIDogYXJncy5zZWVkXG4gICAgfSk7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDbGFzc05hbWUoKTogc3RyaW5nIHtcbiAgICAvLyBJbiBQeXRob24gS2VyYXMsIEdsb3JvdFVuaWZvcm0gaXMgbm90IGEgY2xhc3MsIGJ1dCBhIGhlbHBlciBtZXRob2RcbiAgICAvLyB0aGF0IGNyZWF0ZXMgYSBWYXJpYW5jZVNjYWxpbmcgb2JqZWN0LiBVc2UgJ1ZhcmlhbmNlU2NhbGluZycgYXNcbiAgICAvLyBjbGFzcyBuYW1lIHRvIGJlIGNvbXBhdGlibGUgd2l0aCB0aGF0LlxuICAgIHJldHVybiBWYXJpYW5jZVNjYWxpbmcuY2xhc3NOYW1lO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR2xvcm90VW5pZm9ybSk7XG5cbmV4cG9ydCBjbGFzcyBHbG9yb3ROb3JtYWwgZXh0ZW5kcyBWYXJpYW5jZVNjYWxpbmcge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGNsYXNzTmFtZSA9ICdHbG9yb3ROb3JtYWwnO1xuXG4gIC8qKlxuICAgKiBDb25zdHJ1Y3RvciBvZiBHbG9yb3ROb3JtYWwuXG4gICAqIEBwYXJhbSBzY2FsZVxuICAgKiBAcGFyYW0gbW9kZVxuICAgKiBAcGFyYW0gZGlzdHJpYnV0aW9uXG4gICAqIEBwYXJhbSBzZWVkXG4gICAqL1xuICBjb25zdHJ1Y3RvcihhcmdzPzogU2VlZE9ubHlJbml0aWFsaXplckFyZ3MpIHtcbiAgICBzdXBlcih7XG4gICAgICBzY2FsZTogMS4wLFxuICAgICAgbW9kZTogJ2ZhbkF2ZycsXG4gICAgICBkaXN0cmlidXRpb246ICdub3JtYWwnLFxuICAgICAgc2VlZDogYXJncyA9PSBudWxsID8gbnVsbCA6IGFyZ3Muc2VlZFxuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q2xhc3NOYW1lKCk6IHN0cmluZyB7XG4gICAgLy8gSW4gUHl0aG9uIEtlcmFzLCBHbG9yb3ROb3JtYWwgaXMgbm90IGEgY2xhc3MsIGJ1dCBhIGhlbHBlciBtZXRob2RcbiAgICAvLyB0aGF0IGNyZWF0ZXMgYSBWYXJpYW5jZVNjYWxpbmcgb2JqZWN0LiBVc2UgJ1ZhcmlhbmNlU2NhbGluZycgYXNcbiAgICAvLyBjbGFzcyBuYW1lIHRvIGJlIGNvbXBhdGlibGUgd2l0aCB0aGF0LlxuICAgIHJldHVybiBWYXJpYW5jZVNjYWxpbmcuY2xhc3NOYW1lO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoR2xvcm90Tm9ybWFsKTtcblxuZXhwb3J0IGNsYXNzIEhlTm9ybWFsIGV4dGVuZHMgVmFyaWFuY2VTY2FsaW5nIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBjbGFzc05hbWUgPSAnSGVOb3JtYWwnO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBTZWVkT25seUluaXRpYWxpemVyQXJncykge1xuICAgIHN1cGVyKHtcbiAgICAgIHNjYWxlOiAyLjAsXG4gICAgICBtb2RlOiAnZmFuSW4nLFxuICAgICAgZGlzdHJpYnV0aW9uOiAnbm9ybWFsJyxcbiAgICAgIHNlZWQ6IGFyZ3MgPT0gbnVsbCA/IG51bGwgOiBhcmdzLnNlZWRcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENsYXNzTmFtZSgpOiBzdHJpbmcge1xuICAgIC8vIEluIFB5dGhvbiBLZXJhcywgSGVOb3JtYWwgaXMgbm90IGEgY2xhc3MsIGJ1dCBhIGhlbHBlciBtZXRob2RcbiAgICAvLyB0aGF0IGNyZWF0ZXMgYSBWYXJpYW5jZVNjYWxpbmcgb2JqZWN0LiBVc2UgJ1ZhcmlhbmNlU2NhbGluZycgYXNcbiAgICAvLyBjbGFzcyBuYW1lIHRvIGJlIGNvbXBhdGlibGUgd2l0aCB0aGF0LlxuICAgIHJldHVybiBWYXJpYW5jZVNjYWxpbmcuY2xhc3NOYW1lO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoSGVOb3JtYWwpO1xuXG5leHBvcnQgY2xhc3MgSGVVbmlmb3JtIGV4dGVuZHMgVmFyaWFuY2VTY2FsaW5nIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBjbGFzc05hbWUgPSAnSGVVbmlmb3JtJztcblxuICBjb25zdHJ1Y3RvcihhcmdzPzogU2VlZE9ubHlJbml0aWFsaXplckFyZ3MpIHtcbiAgICBzdXBlcih7XG4gICAgICBzY2FsZTogMi4wLFxuICAgICAgbW9kZTogJ2ZhbkluJyxcbiAgICAgIGRpc3RyaWJ1dGlvbjogJ3VuaWZvcm0nLFxuICAgICAgc2VlZDogYXJncyA9PSBudWxsID8gbnVsbCA6IGFyZ3Muc2VlZFxuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q2xhc3NOYW1lKCk6IHN0cmluZyB7XG4gICAgLy8gSW4gUHl0aG9uIEtlcmFzLCBIZVVuaWZvcm0gaXMgbm90IGEgY2xhc3MsIGJ1dCBhIGhlbHBlciBtZXRob2RcbiAgICAvLyB0aGF0IGNyZWF0ZXMgYSBWYXJpYW5jZVNjYWxpbmcgb2JqZWN0LiBVc2UgJ1ZhcmlhbmNlU2NhbGluZycgYXNcbiAgICAvLyBjbGFzcyBuYW1lIHRvIGJlIGNvbXBhdGlibGUgd2l0aCB0aGF0LlxuICAgIHJldHVybiBWYXJpYW5jZVNjYWxpbmcuY2xhc3NOYW1lO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoSGVVbmlmb3JtKTtcblxuZXhwb3J0IGNsYXNzIExlQ3VuTm9ybWFsIGV4dGVuZHMgVmFyaWFuY2VTY2FsaW5nIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBjbGFzc05hbWUgPSAnTGVDdW5Ob3JtYWwnO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBTZWVkT25seUluaXRpYWxpemVyQXJncykge1xuICAgIHN1cGVyKHtcbiAgICAgIHNjYWxlOiAxLjAsXG4gICAgICBtb2RlOiAnZmFuSW4nLFxuICAgICAgZGlzdHJpYnV0aW9uOiAnbm9ybWFsJyxcbiAgICAgIHNlZWQ6IGFyZ3MgPT0gbnVsbCA/IG51bGwgOiBhcmdzLnNlZWRcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENsYXNzTmFtZSgpOiBzdHJpbmcge1xuICAgIC8vIEluIFB5dGhvbiBLZXJhcywgTGVDdW5Ob3JtYWwgaXMgbm90IGEgY2xhc3MsIGJ1dCBhIGhlbHBlciBtZXRob2RcbiAgICAvLyB0aGF0IGNyZWF0ZXMgYSBWYXJpYW5jZVNjYWxpbmcgb2JqZWN0LiBVc2UgJ1ZhcmlhbmNlU2NhbGluZycgYXNcbiAgICAvLyBjbGFzcyBuYW1lIHRvIGJlIGNvbXBhdGlibGUgd2l0aCB0aGF0LlxuICAgIHJldHVybiBWYXJpYW5jZVNjYWxpbmcuY2xhc3NOYW1lO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoTGVDdW5Ob3JtYWwpO1xuXG5leHBvcnQgY2xhc3MgTGVDdW5Vbmlmb3JtIGV4dGVuZHMgVmFyaWFuY2VTY2FsaW5nIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBvdmVycmlkZSBjbGFzc05hbWUgPSAnTGVDdW5Vbmlmb3JtJztcblxuICBjb25zdHJ1Y3RvcihhcmdzPzogU2VlZE9ubHlJbml0aWFsaXplckFyZ3MpIHtcbiAgICBzdXBlcih7XG4gICAgICBzY2FsZTogMS4wLFxuICAgICAgbW9kZTogJ2ZhbkluJyxcbiAgICAgIGRpc3RyaWJ1dGlvbjogJ3VuaWZvcm0nLFxuICAgICAgc2VlZDogYXJncyA9PSBudWxsID8gbnVsbCA6IGFyZ3Muc2VlZFxuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q2xhc3NOYW1lKCk6IHN0cmluZyB7XG4gICAgLy8gSW4gUHl0aG9uIEtlcmFzLCBMZUN1blVuaWZvcm0gaXMgbm90IGEgY2xhc3MsIGJ1dCBhIGhlbHBlciBtZXRob2RcbiAgICAvLyB0aGF0IGNyZWF0ZXMgYSBWYXJpYW5jZVNjYWxpbmcgb2JqZWN0LiBVc2UgJ1ZhcmlhbmNlU2NhbGluZycgYXNcbiAgICAvLyBjbGFzcyBuYW1lIHRvIGJlIGNvbXBhdGlibGUgd2l0aCB0aGF0LlxuICAgIHJldHVybiBWYXJpYW5jZVNjYWxpbmcuY2xhc3NOYW1lO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoTGVDdW5Vbmlmb3JtKTtcblxuZXhwb3J0IGludGVyZmFjZSBPcnRob2dvbmFsQXJncyBleHRlbmRzIFNlZWRPbmx5SW5pdGlhbGl6ZXJBcmdzIHtcbiAgLyoqXG4gICAqIE11bHRpcGxpY2F0aXZlIGZhY3RvciB0byBhcHBseSB0byB0aGUgb3J0aG9nb25hbCBtYXRyaXguIERlZmF1bHRzIHRvIDEuXG4gICAqL1xuICBnYWluPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgT3J0aG9nb25hbCBleHRlbmRzIEluaXRpYWxpemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnT3J0aG9nb25hbCc7XG4gIHJlYWRvbmx5IERFRkFVTFRfR0FJTiA9IDE7XG4gIHByb3RlY3RlZCByZWFkb25seSBnYWluOiBudW1iZXI7XG4gIHByb3RlY3RlZCByZWFkb25seSBzZWVkOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoYXJncz86IE9ydGhvZ29uYWxBcmdzKSB7XG4gICAgc3VwZXIoKTtcbiAgICB0aGlzLmdhaW4gPSBhcmdzLmdhaW4gPT0gbnVsbCA/IHRoaXMuREVGQVVMVF9HQUlOIDogYXJncy5nYWluO1xuICAgIHRoaXMuc2VlZCA9IGFyZ3Muc2VlZDtcblxuICAgIGlmICh0aGlzLnNlZWQgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgJ1JhbmRvbSBzZWVkIGlzIG5vdCBpbXBsZW1lbnRlZCBmb3IgT3J0aG9nb25hbCBJbml0aWFsaXplciB5ZXQuJyk7XG4gICAgfVxuICB9XG5cbiAgYXBwbHkoc2hhcGU6IFNoYXBlLCBkdHlwZT86IERhdGFUeXBlKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpZiAoc2hhcGUubGVuZ3RoIDwgMikge1xuICAgICAgICB0aHJvdyBuZXcgTm90SW1wbGVtZW50ZWRFcnJvcignU2hhcGUgbXVzdCBiZSBhdCBsZWFzdCAyRC4nKTtcbiAgICAgIH1cbiAgICAgIGlmIChzaGFwZVswXSAqIHNoYXBlWzFdID4gMjAwMCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgT3J0aG9nb25hbCBpbml0aWFsaXplciBpcyBiZWluZyBjYWxsZWQgb24gYSBtYXRyaXggd2l0aCBtb3JlIGAgK1xuICAgICAgICAgICAgYHRoYW4gMjAwMCAoJHtzaGFwZVswXSAqIHNoYXBlWzFdfSkgZWxlbWVudHM6IGAgK1xuICAgICAgICAgICAgYFNsb3duZXNzIG1heSByZXN1bHQuYCk7XG4gICAgICB9XG5cbiAgICAgIC8vIFRPRE8oY2Fpcyk6IEFkZCBzZWVkIHN1cHBvcnQuXG4gICAgICBjb25zdCBub3JtYWxpemVkU2hhcGUgPVxuICAgICAgICAgIHNoYXBlWzBdID4gc2hhcGVbMV0gPyBbc2hhcGVbMV0sIHNoYXBlWzBdXSA6IHNoYXBlO1xuICAgICAgY29uc3QgYSA9IEsucmFuZG9tTm9ybWFsKG5vcm1hbGl6ZWRTaGFwZSwgMCwgMSwgJ2Zsb2F0MzInKSBhcyBUZW5zb3IyRDtcbiAgICAgIGxldCBxID0gbGluYWxnLmdyYW1TY2htaWR0KGEpIGFzIFRlbnNvcjJEO1xuICAgICAgaWYgKHNoYXBlWzBdID4gc2hhcGVbMV0pIHtcbiAgICAgICAgcSA9IHRyYW5zcG9zZShxKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBtdWwodGhpcy5nYWluLCBxKTtcbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7XG4gICAgICBnYWluOiB0aGlzLmdhaW4sXG4gICAgICBzZWVkOiB0aGlzLnNlZWQsXG4gICAgfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE9ydGhvZ29uYWwpO1xuXG4vKiogQGRvY2lubGluZSAqL1xuZXhwb3J0IHR5cGUgSW5pdGlhbGl6ZXJJZGVudGlmaWVyID1cbiAgICAnY29uc3RhbnQnfCdnbG9yb3ROb3JtYWwnfCdnbG9yb3RVbmlmb3JtJ3wnaGVOb3JtYWwnfCdoZVVuaWZvcm0nfCdpZGVudGl0eSd8XG4gICAgJ2xlQ3VuTm9ybWFsJ3wnbGVDdW5Vbmlmb3JtJ3wnb25lcyd8J29ydGhvZ29uYWwnfCdyYW5kb21Ob3JtYWwnfFxuICAgICdyYW5kb21Vbmlmb3JtJ3wndHJ1bmNhdGVkTm9ybWFsJ3wndmFyaWFuY2VTY2FsaW5nJ3wnemVyb3MnfHN0cmluZztcblxuLy8gTWFwcyB0aGUgSmF2YVNjcmlwdC1saWtlIGlkZW50aWZpZXIga2V5cyB0byB0aGUgY29ycmVzcG9uZGluZyByZWdpc3RyeVxuLy8gc3ltYm9scy5cbmV4cG9ydCBjb25zdCBJTklUSUFMSVpFUl9JREVOVElGSUVSX1JFR0lTVFJZX1NZTUJPTF9NQVA6XG4gICAge1tpZGVudGlmaWVyIGluIEluaXRpYWxpemVySWRlbnRpZmllcl06IHN0cmluZ30gPSB7XG4gICAgICAnY29uc3RhbnQnOiAnQ29uc3RhbnQnLFxuICAgICAgJ2dsb3JvdE5vcm1hbCc6ICdHbG9yb3ROb3JtYWwnLFxuICAgICAgJ2dsb3JvdFVuaWZvcm0nOiAnR2xvcm90VW5pZm9ybScsXG4gICAgICAnaGVOb3JtYWwnOiAnSGVOb3JtYWwnLFxuICAgICAgJ2hlVW5pZm9ybSc6ICdIZVVuaWZvcm0nLFxuICAgICAgJ2lkZW50aXR5JzogJ0lkZW50aXR5JyxcbiAgICAgICdsZUN1bk5vcm1hbCc6ICdMZUN1bk5vcm1hbCcsXG4gICAgICAnbGVDdW5Vbmlmb3JtJzogJ0xlQ3VuVW5pZm9ybScsXG4gICAgICAnb25lcyc6ICdPbmVzJyxcbiAgICAgICdvcnRob2dvbmFsJzogJ09ydGhvZ29uYWwnLFxuICAgICAgJ3JhbmRvbU5vcm1hbCc6ICdSYW5kb21Ob3JtYWwnLFxuICAgICAgJ3JhbmRvbVVuaWZvcm0nOiAnUmFuZG9tVW5pZm9ybScsXG4gICAgICAndHJ1bmNhdGVkTm9ybWFsJzogJ1RydW5jYXRlZE5vcm1hbCcsXG4gICAgICAndmFyaWFuY2VTY2FsaW5nJzogJ1ZhcmlhbmNlU2NhbGluZycsXG4gICAgICAnemVyb3MnOiAnWmVyb3MnXG4gICAgfTtcblxuZnVuY3Rpb24gZGVzZXJpYWxpemVJbml0aWFsaXplcihcbiAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICBjdXN0b21PYmplY3RzOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7fSk6IEluaXRpYWxpemVyIHtcbiAgcmV0dXJuIGRlc2VyaWFsaXplS2VyYXNPYmplY3QoXG4gICAgICBjb25maWcsIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphdGlvbk1hcC5nZXRNYXAoKS5jbGFzc05hbWVNYXAsXG4gICAgICBjdXN0b21PYmplY3RzLCAnaW5pdGlhbGl6ZXInKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNlcmlhbGl6ZUluaXRpYWxpemVyKGluaXRpYWxpemVyOiBJbml0aWFsaXplcik6XG4gICAgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0VmFsdWUge1xuICByZXR1cm4gc2VyaWFsaXplS2VyYXNPYmplY3QoaW5pdGlhbGl6ZXIpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0SW5pdGlhbGl6ZXIoaWRlbnRpZmllcjogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCk6IEluaXRpYWxpemVyIHtcbiAgaWYgKHR5cGVvZiBpZGVudGlmaWVyID09PSAnc3RyaW5nJykge1xuICAgIGNvbnN0IGNsYXNzTmFtZSA9IGlkZW50aWZpZXIgaW4gSU5JVElBTElaRVJfSURFTlRJRklFUl9SRUdJU1RSWV9TWU1CT0xfTUFQID9cbiAgICAgICAgSU5JVElBTElaRVJfSURFTlRJRklFUl9SRUdJU1RSWV9TWU1CT0xfTUFQW2lkZW50aWZpZXJdIDpcbiAgICAgICAgaWRlbnRpZmllcjtcbiAgICAvKiBXZSBoYXZlIGZvdXIgJ2hlbHBlcicgY2xhc3NlcyBmb3IgY29tbW9uIGluaXRpYWxpemVycyB0aGF0XG4gICAgYWxsIGdldCBzZXJpYWxpemVkIGFzICdWYXJpYW5jZVNjYWxpbmcnIGFuZCBzaG91bGRuJ3QgZ28gdGhyb3VnaFxuICAgIHRoZSBkZXNlcmlhbGl6ZUluaXRpYWxpemVyIHBhdGh3YXkuICovXG4gICAgaWYgKGNsYXNzTmFtZSA9PT0gJ0dsb3JvdE5vcm1hbCcpIHtcbiAgICAgIHJldHVybiBuZXcgR2xvcm90Tm9ybWFsKCk7XG4gICAgfSBlbHNlIGlmIChjbGFzc05hbWUgPT09ICdHbG9yb3RVbmlmb3JtJykge1xuICAgICAgcmV0dXJuIG5ldyBHbG9yb3RVbmlmb3JtKCk7XG4gICAgfSBlbHNlIGlmIChjbGFzc05hbWUgPT09ICdIZU5vcm1hbCcpIHtcbiAgICAgIHJldHVybiBuZXcgSGVOb3JtYWwoKTtcbiAgICB9IGVsc2UgaWYgKGNsYXNzTmFtZSA9PT0gJ0hlVW5pZm9ybScpIHtcbiAgICAgIHJldHVybiBuZXcgSGVVbmlmb3JtKCk7XG4gICAgfSBlbHNlIGlmIChjbGFzc05hbWUgPT09ICdMZUN1bk5vcm1hbCcpIHtcbiAgICAgIHJldHVybiBuZXcgTGVDdW5Ob3JtYWwoKTtcbiAgICB9IGVsc2UgaWYgKGNsYXNzTmFtZSA9PT0gJ0xlQ3VuVW5pZm9ybScpIHtcbiAgICAgIHJldHVybiBuZXcgTGVDdW5Vbmlmb3JtKCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge307XG4gICAgICBjb25maWdbJ2NsYXNzTmFtZSddID0gY2xhc3NOYW1lO1xuICAgICAgY29uZmlnWydjb25maWcnXSA9IHt9O1xuICAgICAgcmV0dXJuIGRlc2VyaWFsaXplSW5pdGlhbGl6ZXIoY29uZmlnKTtcbiAgICB9XG4gIH0gZWxzZSBpZiAoaWRlbnRpZmllciBpbnN0YW5jZW9mIEluaXRpYWxpemVyKSB7XG4gICAgcmV0dXJuIGlkZW50aWZpZXI7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIGRlc2VyaWFsaXplSW5pdGlhbGl6ZXIoaWRlbnRpZmllcik7XG4gIH1cbn1cbiJdfQ==