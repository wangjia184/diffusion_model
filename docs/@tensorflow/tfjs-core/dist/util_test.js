/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from './index';
import { ALL_ENVS, describeWithFlags } from './jasmine_util';
import { complex, scalar, tensor2d } from './ops/ops';
import { inferShape } from './tensor_util_env';
import * as util from './util';
describe('Util', () => {
    it('Correctly gets size from shape', () => {
        expect(util.sizeFromShape([1, 2, 3, 4])).toEqual(24);
    });
    it('Correctly identifies scalars', () => {
        expect(util.isScalarShape([])).toBe(true);
        expect(util.isScalarShape([1, 2])).toBe(false);
        expect(util.isScalarShape([1])).toBe(false);
    });
    it('Number arrays equal', () => {
        expect(util.arraysEqual([1, 2, 3, 6], [1, 2, 3, 6])).toBe(true);
        expect(util.arraysEqual([1, 2], [1, 2, 3])).toBe(false);
        expect(util.arraysEqual([1, 2, 5], [1, 2])).toBe(false);
    });
    it('Arrays shuffle randomly', () => {
        // Create 1000 numbers ordered
        const a = Array.apply(0, { length: 1000 }).map(Number.call, Number).slice(1);
        const b = [].concat(a); // copy ES5 style
        util.shuffle(a);
        expect(a).not.toEqual(b);
        expect(a.length).toEqual(b.length);
    });
    it('Multiple arrays shuffle together', () => {
        // Create 1000 numbers ordered
        const a = Array.apply(0, { length: 1000 }).map(Number.call, Number).slice(1);
        const b = [].concat(a); // copies
        const c = [].concat(a);
        util.shuffleCombo(a, b);
        expect(a).not.toEqual(c);
        expect(a).toEqual(b);
        expect(a.length).toEqual(c.length);
    });
    it('Is integer', () => {
        expect(util.isInt(0.5)).toBe(false);
        expect(util.isInt(1)).toBe(true);
    });
    it('Size to squarish shape (perfect square)', () => {
        expect(util.sizeToSquarishShape(9)).toEqual([3, 3]);
    });
    it('Size to squarish shape (prime number)', () => {
        expect(util.sizeToSquarishShape(11)).toEqual([4, 3]);
    });
    it('Size to squarish shape (almost square)', () => {
        expect(util.sizeToSquarishShape(35)).toEqual([6, 6]);
    });
    it('Size of 1 to squarish shape', () => {
        expect(util.sizeToSquarishShape(1)).toEqual([1, 1]);
    });
    it('infer shape single number', () => {
        expect(inferShape(4)).toEqual([]);
    });
    it('infer shape 1d array', () => {
        expect(inferShape([1, 2, 5])).toEqual([3]);
    });
    it('infer shape 2d array', () => {
        expect(inferShape([[1, 2, 5], [5, 4, 1]])).toEqual([2, 3]);
    });
    it('infer shape 3d array', () => {
        const a = [[[1, 2], [2, 3], [5, 6]], [[5, 6], [4, 5], [1, 2]]];
        expect(inferShape(a)).toEqual([2, 3, 2]);
    });
    it('infer shape 4d array', () => {
        const a = [
            [[[1], [2]], [[2], [3]], [[5], [6]]], [[[5], [6]], [[4], [5]], [[1], [2]]]
        ];
        expect(inferShape(a)).toEqual([2, 3, 2, 1]);
    });
    it('infer shape of typed array', () => {
        const a = new Float32Array([1, 2, 3, 4, 5]);
        expect(inferShape(a)).toEqual([5]);
    });
    it('infer shape of clamped typed array', () => {
        const a = new Uint8ClampedArray([1, 2, 3, 4, 5]);
        expect(inferShape(a)).toEqual([5]);
    });
    it('infer shape of Uint8Array[], string tensor', () => {
        const a = [new Uint8Array([1, 2]), new Uint8Array([3, 4])];
        expect(inferShape(a, 'string')).toEqual([2]);
    });
    it('infer shape of Uint8Array[][], string tensor', () => {
        const a = [
            [new Uint8Array([1]), new Uint8Array([2])],
            [new Uint8Array([1]), new Uint8Array([2])]
        ];
        expect(inferShape(a, 'string')).toEqual([2, 2]);
    });
    it('infer shape of Uint8Array[][][], string tensor', () => {
        const a = [
            [[new Uint8Array([1, 2])], [new Uint8Array([2, 1])]],
            [[new Uint8Array([1, 2])], [new Uint8Array([2, 1])]]
        ];
        expect(inferShape(a, 'string')).toEqual([2, 2, 1]);
    });
});
describe('util.flatten', () => {
    it('empty', () => {
        const data = [];
        expect(util.flatten(data)).toEqual([]);
    });
    it('nested number arrays', () => {
        expect(util.flatten([[1, 2, 3], [4, 5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
        expect(util.flatten([[[1, 2], [3, 4], [5, 6], [7, 8]]])).toEqual([
            1, 2, 3, 4, 5, 6, 7, 8
        ]);
        expect(util.flatten([1, 2, 3, 4, 5, 6])).toEqual([1, 2, 3, 4, 5, 6]);
    });
    it('nested string arrays', () => {
        expect(util.flatten([['a', 'b'], ['c', [['d']]]])).toEqual([
            'a', 'b', 'c', 'd'
        ]);
        expect(util.flatten([['a', ['b']], ['c', [['d']], 'e']])).toEqual([
            'a', 'b', 'c', 'd', 'e'
        ]);
    });
    it('mixed TypedArray and number[]', () => {
        const data = [new Float32Array([1, 2]), 3, [4, 5, new Float32Array([6, 7])]];
        expect(util.flatten(data)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    });
    it('nested Uint8Arrays, skipTypedArray=true', () => {
        const data = [
            [new Uint8Array([1, 2]), new Uint8Array([3, 4])],
            [new Uint8Array([5, 6]), new Uint8Array([7, 8])]
        ];
        expect(util.flatten(data, [], true)).toEqual([
            new Uint8Array([1, 2]), new Uint8Array([3, 4]), new Uint8Array([5, 6]),
            new Uint8Array([7, 8])
        ]);
    });
    it('Int8Array', () => {
        const data = [new Int8Array([1, 2])];
        expect(util.flatten(data)).toEqual([1, 2]);
    });
    it('index signature', () => {
        const data = { 0: 1, 1: 2 };
        // Will be ignored since array iteration ignores negatives.
        data[-1] = -1;
        // Will be ignored since non-integer array keys are ignored.
        data[3.2] = 4;
        expect(util.flatten(data)).toEqual([1, 2]);
    });
});
function encodeStrings(a) {
    return a.map(s => util.encodeString(s));
}
describe('util.bytesFromStringArray', () => {
    it('count bytes after utf8 encoding', () => {
        expect(util.bytesFromStringArray(encodeStrings(['a', 'bb', 'ccc'])))
            .toBe(6);
        expect(util.bytesFromStringArray(encodeStrings(['a', 'bb', 'cccddd'])))
            .toBe(9);
        expect(util.bytesFromStringArray(encodeStrings(['даниел']))).toBe(6 * 2);
    });
});
describe('util.inferDtype', () => {
    it('a single string => string', () => {
        expect(util.inferDtype('hello')).toBe('string');
    });
    it('a single boolean => bool', () => {
        expect(util.inferDtype(true)).toBe('bool');
        expect(util.inferDtype(false)).toBe('bool');
    });
    it('a single number => float32', () => {
        expect(util.inferDtype(0)).toBe('float32');
        expect(util.inferDtype(34)).toBe('float32');
    });
    it('a list of strings => string', () => {
        // Flat.
        expect(util.inferDtype(['a', 'b', 'c'])).toBe('string');
        // Nested.
        expect(util.inferDtype([
            [['a']], [['b']], [['c']], [['d']]
        ])).toBe('string');
    });
    it('a list of bools => float32', () => {
        // Flat.
        expect(util.inferDtype([false, true, false])).toBe('bool');
        // Nested.
        expect(util.inferDtype([
            [[true]], [[false]], [[true]], [[true]]
        ])).toBe('bool');
    });
    it('a list of numbers => float32', () => {
        // Flat.
        expect(util.inferDtype([0, 1, 2])).toBe('float32');
        // Nested.
        expect(util.inferDtype([[[0]], [[1]], [[2]], [[3]]])).toBe('float32');
    });
});
describe('util.repeatedTry', () => {
    it('resolves', (doneFn) => {
        let counter = 0;
        const checkFn = () => {
            counter++;
            if (counter === 2) {
                return true;
            }
            return false;
        };
        util.repeatedTry(checkFn).then(doneFn).catch(() => {
            throw new Error('Rejected backoff.');
        });
    });
    it('rejects', (doneFn) => {
        const checkFn = () => false;
        util.repeatedTry(checkFn, () => 0, 5)
            .then(() => {
            throw new Error('Backoff resolved');
        })
            .catch(doneFn);
    });
});
describe('util.inferFromImplicitShape', () => {
    it('empty shape', () => {
        const result = util.inferFromImplicitShape([], 0);
        expect(result).toEqual([]);
    });
    it('[2, 3, 4] -> [2, 3, 4]', () => {
        const result = util.inferFromImplicitShape([2, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, 4] -> [2, 3, 4], size=24', () => {
        const result = util.inferFromImplicitShape([2, -1, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[-1, 3, 4] -> [2, 3, 4], size=24', () => {
        const result = util.inferFromImplicitShape([-1, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, 3, -1] -> [2, 3, 4], size=24', () => {
        const result = util.inferFromImplicitShape([2, 3, -1], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, -1] throws error', () => {
        expect(() => util.inferFromImplicitShape([2, -1, -1], 24)).toThrowError();
    });
    it('[2, 3, -1] size=13 throws error', () => {
        expect(() => util.inferFromImplicitShape([2, 3, -1], 13)).toThrowError();
    });
    it('[2, 3, 4] size=25 (should be 24) throws error', () => {
        expect(() => util.inferFromImplicitShape([2, 3, 4], 25)).toThrowError();
    });
});
describe('util parseAxisParam', () => {
    it('axis=null returns no axes for scalar', () => {
        const axis = null;
        const shape = [];
        expect(util.parseAxisParam(axis, shape)).toEqual([]);
    });
    it('axis=null returns 0 axis for Tensor1D', () => {
        const axis = null;
        const shape = [4];
        expect(util.parseAxisParam(axis, shape)).toEqual([0]);
    });
    it('axis=null returns all axes for Tensor3D', () => {
        const axis = null;
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 1, 2]);
    });
    it('axis as a single number', () => {
        const axis = 1;
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([1]);
    });
    it('axis as single negative number', () => {
        const axis = -1;
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([2]);
        const axis2 = -2;
        expect(util.parseAxisParam(axis2, shape)).toEqual([1]);
        const axis3 = -3;
        expect(util.parseAxisParam(axis3, shape)).toEqual([0]);
    });
    it('axis as list of negative numbers', () => {
        const axis = [-1, -3];
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([2, 0]);
    });
    it('axis as list of positive numbers', () => {
        const axis = [0, 2];
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
    });
    it('axis as combo of positive and negative numbers', () => {
        const axis = [0, -1];
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
    });
    it('axis out of range throws error', () => {
        const axis = -4;
        const shape = [3, 1, 2];
        expect(() => util.parseAxisParam(axis, shape)).toThrowError();
        const axis2 = 4;
        expect(() => util.parseAxisParam(axis2, shape)).toThrowError();
    });
    it('axis a list with one number out of range throws error', () => {
        const axis = [0, 4];
        const shape = [3, 1, 2];
        expect(() => util.parseAxisParam(axis, shape)).toThrowError();
    });
    it('axis with decimal value throws error', () => {
        const axis = 0.5;
        const shape = [3, 1, 2];
        expect(() => util.parseAxisParam(axis, shape)).toThrowError();
    });
});
describe('util.squeezeShape', () => {
    it('scalar', () => {
        const { newShape, keptDims } = util.squeezeShape([]);
        expect(newShape).toEqual([]);
        expect(keptDims).toEqual([]);
    });
    it('1x1 reduced to scalar', () => {
        const { newShape, keptDims } = util.squeezeShape([1, 1]);
        expect(newShape).toEqual([]);
        expect(keptDims).toEqual([]);
    });
    it('1x3x1 reduced to [3]', () => {
        const { newShape, keptDims } = util.squeezeShape([1, 3, 1]);
        expect(newShape).toEqual([3]);
        expect(keptDims).toEqual([1]);
    });
    it('1x1x4 reduced to [4]', () => {
        const { newShape, keptDims } = util.squeezeShape([1, 1, 4]);
        expect(newShape).toEqual([4]);
        expect(keptDims).toEqual([2]);
    });
    it('2x3x4 not reduction', () => {
        const { newShape, keptDims } = util.squeezeShape([2, 3, 4]);
        expect(newShape).toEqual([2, 3, 4]);
        expect(keptDims).toEqual([0, 1, 2]);
    });
    describe('with axis', () => {
        it('should only reduce dimensions specified by axis', () => {
            const { newShape, keptDims } = util.squeezeShape([1, 1, 1, 1, 4], [1, 2]);
            expect(newShape).toEqual([1, 1, 4]);
            expect(keptDims).toEqual([0, 3, 4]);
        });
        it('should only reduce dimensions specified by negative axis', () => {
            const { newShape, keptDims } = util.squeezeShape([1, 1, 1, 1, 4], [-2, -3]);
            expect(newShape).toEqual([1, 1, 4]);
            expect(keptDims).toEqual([0, 1, 4]);
        });
        it('should only reduce dimensions specified by negative axis', () => {
            const axis = [-2, -3];
            util.squeezeShape([1, 1, 1, 1, 4], axis);
            expect(axis).toEqual([-2, -3]);
        });
        it('throws error when specified axis is not squeezable', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [1, 2])).toThrowError();
        });
        it('throws error when specified negative axis is not squeezable', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [-1, -2])).toThrowError();
        });
        it('throws error when specified axis is out of range', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [11, 22])).toThrowError();
        });
        it('throws error when specified negative axis is out of range', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [
                -11, -22
            ])).toThrowError();
        });
    });
});
describe('util.checkConversionForErrors', () => {
    it('Float32Array has NaN', () => {
        expect(() => util.checkConversionForErrors(new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32'))
            .toThrowError();
    });
    it('Float32Array has Infinity', () => {
        expect(() => util.checkConversionForErrors(new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32'))
            .toThrowError();
    });
    it('Int32Array has NaN', () => {
        expect(() => util.checkConversionForErrors([1, 2, 3, 4, NaN], 'int32'))
            .toThrowError();
    });
});
describe('util.hasEncodingLoss', () => {
    it('complex64 to any', () => {
        expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
        expect(util.hasEncodingLoss('complex64', 'int32')).toBe(true);
        expect(util.hasEncodingLoss('complex64', 'bool')).toBe(true);
    });
    it('any to complex64', () => {
        expect(util.hasEncodingLoss('bool', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
    });
    it('any to float32', () => {
        expect(util.hasEncodingLoss('bool', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
    });
    it('float32 to any', () => {
        expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'int32')).toBe(true);
        expect(util.hasEncodingLoss('float32', 'bool')).toBe(true);
        expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
    });
    it('int32 to lower', () => {
        expect(util.hasEncodingLoss('int32', 'int32')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'bool')).toBe(true);
    });
    it('lower to int32', () => {
        expect(util.hasEncodingLoss('bool', 'int32')).toBe(false);
    });
    it('bool to bool', () => {
        expect(util.hasEncodingLoss('bool', 'bool')).toBe(false);
    });
});
describeWithFlags('util.toNestedArray', ALL_ENVS, () => {
    it('2 dimensions', () => {
        const a = new Float32Array([1, 2, 3, 4, 5, 6]);
        expect(util.toNestedArray([2, 3], a)).toEqual([[1, 2, 3], [4, 5, 6]]);
    });
    it('3 dimensions (2x2x3)', () => {
        const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(util.toNestedArray([2, 2, 3], a)).toEqual([
            [[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]
        ]);
    });
    it('3 dimensions (3x2x2)', () => {
        const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(util.toNestedArray([3, 2, 2], a)).toEqual([
            [[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]
        ]);
    });
    it('invalid dimension', () => {
        const a = new Float32Array([1, 2, 3]);
        expect(() => util.toNestedArray([2, 2], a)).toThrowError();
    });
    it('tensor to nested array', async () => {
        const x = tensor2d([1, 2, 3, 4], [2, 2]);
        expect(util.toNestedArray(x.shape, await x.data())).toEqual([
            [1, 2], [3, 4]
        ]);
    });
    it('scalar to nested array', async () => {
        const x = scalar(1);
        expect(util.toNestedArray(x.shape, await x.data())).toEqual(1);
    });
    it('tensor with zero shape', () => {
        const a = new Float32Array([0, 1]);
        expect(util.toNestedArray([1, 0, 2], a)).toEqual([]);
    });
});
describeWithFlags('util.toNestedArray for a complex tensor', ALL_ENVS, () => {
    it('2 dimensions', () => {
        const a = new Float32Array([1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16]);
        expect(util.toNestedArray([2, 3], a, true)).toEqual([
            [1, 11, 2, 12, 3, 13], [4, 14, 5, 15, 6, 16]
        ]);
    });
    it('3 dimensions (2x2x3)', () => {
        const a = new Float32Array([
            0, 50, 1, 51, 2, 52, 3, 53, 4, 54, 5, 55,
            6, 56, 7, 57, 8, 58, 9, 59, 10, 60, 11, 61
        ]);
        expect(util.toNestedArray([2, 2, 3], a, true)).toEqual([
            [[0, 50, 1, 51, 2, 52], [3, 53, 4, 54, 5, 55]],
            [[6, 56, 7, 57, 8, 58], [9, 59, 10, 60, 11, 61]]
        ]);
    });
    it('3 dimensions (3x2x2)', () => {
        const a = new Float32Array([
            0, 50, 1, 51, 2, 52, 3, 53, 4, 54, 5, 55,
            6, 56, 7, 57, 8, 58, 9, 59, 10, 60, 11, 61
        ]);
        expect(util.toNestedArray([3, 2, 2], a, true)).toEqual([
            [[0, 50, 1, 51], [2, 52, 3, 53]], [[4, 54, 5, 55], [6, 56, 7, 57]],
            [[8, 58, 9, 59], [10, 60, 11, 61]]
        ]);
    });
    it('invalid dimension', () => {
        const a = new Float32Array([1, 11, 2, 12, 3, 13]);
        expect(() => util.toNestedArray([2, 2], a, true)).toThrowError();
    });
    it('tensor to nested array', async () => {
        const x = complex([[1, 2], [3, 4]], [[11, 12], [13, 14]]);
        expect(util.toNestedArray(x.shape, await x.data(), true)).toEqual([
            [1, 11, 2, 12], [3, 13, 4, 14]
        ]);
    });
});
describe('util.fetch', () => {
    it('should call the platform fetch', () => {
        spyOn(tf.env().platform, 'fetch')
            .and.callFake(async () => ({}));
        util.fetch('test/path', { method: 'GET' });
        expect(tf.env().platform.fetch).toHaveBeenCalledWith('test/path', {
            method: 'GET'
        });
    });
});
describe('util.encodeString', () => {
    it('Encode an empty string, default encoding', () => {
        const res = util.encodeString('');
        expect(res).toEqual(new Uint8Array([]));
    });
    it('Encode an empty string, utf-8 encoding', () => {
        const res = util.encodeString('', 'utf-8');
        expect(res).toEqual(new Uint8Array([]));
    });
    it('Encode an empty string, invalid decoding', () => {
        expect(() => util.encodeString('', 'foobarbax')).toThrowError();
    });
    it('Encode cyrillic letters', () => {
        const res = util.encodeString('Kaкo стe');
        expect(res).toEqual(new Uint8Array([75, 97, 208, 186, 111, 32, 209, 129, 209, 130, 101]));
    });
    it('Encode ascii letters', () => {
        const res = util.encodeString('hello');
        expect(res).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
    });
});
describe('util.decodeString', () => {
    it('decode an empty string', () => {
        const s = util.decodeString(new Uint8Array([]));
        expect(s).toEqual('');
    });
    it('decode ascii', () => {
        const s = util.decodeString(new Uint8Array([104, 101, 108, 108, 111]));
        expect(s).toEqual('hello');
    });
    it('decode cyrillic', () => {
        const s = util.decodeString(new Uint8Array([75, 97, 208, 186, 111, 32, 209, 129, 209, 130, 101]));
        expect(s).toEqual('Kaкo стe');
    });
    it('decode utf-16', () => {
        const s = util.decodeString(new Uint8Array([255, 254, 237, 139, 0, 138, 4, 89, 6, 116]), 'utf-16');
        // UTF-16 allows optional presence of byte-order-mark (BOM)
        // Construct string for '语言处理', with and without BOM
        const expected = String.fromCodePoint(0x8bed, 0x8a00, 0x5904, 0x7406);
        const expectedBOM = String.fromCodePoint(0xfeff, 0x8bed, 0x8a00, 0x5904, 0x7406);
        if (s.codePointAt(0) === 0xfeff) {
            expect(s).toEqual(expectedBOM);
        }
        else {
            expect(s).toEqual(expected);
        }
    });
    it('assert promise', () => {
        const promise = new Promise(() => { });
        expect(util.isPromise(promise)).toBeTruthy();
        const promise2 = { then: () => { } };
        expect(util.isPromise(promise2)).toBeTruthy();
        const promise3 = {};
        expect(util.isPromise(promise3)).toBeFalsy();
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy91dGlsX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxLQUFLLEVBQUUsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzNELE9BQU8sRUFBQyxPQUFPLEVBQUUsTUFBTSxFQUFFLFFBQVEsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNwRCxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDN0MsT0FBTyxLQUFLLElBQUksTUFBTSxRQUFRLENBQUM7QUFFL0IsUUFBUSxDQUFDLE1BQU0sRUFBRSxHQUFHLEVBQUU7SUFDcEIsRUFBRSxDQUFDLGdDQUFnQyxFQUFFLEdBQUcsRUFBRTtRQUN4QyxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUM7SUFDdkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsOEJBQThCLEVBQUUsR0FBRyxFQUFFO1FBQ3RDLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDL0MsTUFBTSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUss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QyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDN0MsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO1FBQzlCLE1BQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMvRCxNQUFNLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRztZQUNSLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMs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RSxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDckQsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1lBQzlDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztTQUNqRCxDQUFDLENBQUM7SUFDTCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxzQkFBc0IsRUFBRSxHQUFHLEVBQUU7UUFDOUIsTUFBTSxDQUFDLEdBQUcsSUFBSSxZQUFZLENBQUM7WUFDekIsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUcsRUFBRSxFQUFFLENBQUMsRUFBRyxFQUFFO1lBQzFDLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRTtTQUMzQyxDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDO1lBQ3JELENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFB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Q0wsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCAqIGFzIHRmIGZyb20gJy4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4vamFzbWluZV91dGlsJztcbmltcG9ydCB7Y29tcGxleCwgc2NhbGFyLCB0ZW5zb3IyZH0gZnJvbSAnLi9vcHMvb3BzJztcbmltcG9ydCB7aW5mZXJTaGFwZX0gZnJvbSAnLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuL3V0aWwnO1xuXG5kZXNjcmliZSgnVXRpbCcsICgpID0+IHtcbiAgaXQoJ0NvcnJlY3RseSBnZXRzIHNpemUgZnJvbSBzaGFwZScsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5zaXplRnJvbVNoYXBlKFsxLCAyLCAzLCA0XSkpLnRvRXF1YWwoMjQpO1xuICB9KTtcblxuICBpdCgnQ29ycmVjdGx5IGlkZW50aWZpZXMgc2NhbGFycycsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5pc1NjYWxhclNoYXBlKFtdKSkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodXRpbC5pc1NjYWxhclNoYXBlKFsxLCAyXSkpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmlzU2NhbGFyU2hhcGUoWzFdKSkudG9CZShmYWxzZSk7XG4gIH0pO1xuXG4gIGl0KCdOdW1iZXIgYXJyYXlzIGVxdWFsJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmFycmF5c0VxdWFsKFsxLCAyLCAzLCA2XSwgWzEsIDIsIDMsIDZdKSkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodXRpbC5hcnJheXNFcXVhbChbMSwgMl0sIFsxLCAyLCAzXSkpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmFycmF5c0VxdWFsKFsxLCAyLCA1XSwgWzEsIDJdKSkudG9CZShmYWxzZSk7XG4gIH0pO1xuXG4gIGl0KCdBcnJheXMgc2h1ZmZsZSByYW5kb21seScsICgpID0+IHtcbiAgICAvLyBDcmVhdGUgMTAwMCBudW1iZXJzIG9yZGVyZWRcbiAgICBjb25zdCBhID0gQXJyYXkuYXBwbHkoMCwge2xlbmd0aDogMTAwMH0pLm1hcChOdW1iZXIuY2FsbCwgTnVtYmVyKS5zbGljZSgxKTtcbiAgICBjb25zdCBiID0gW10uY29uY2F0KGEpOyAgLy8gY29weSBFUzUgc3R5bGVcbiAgICB1dGlsLnNodWZmbGUoYSk7XG4gICAgZXhwZWN0KGEpLm5vdC50b0VxdWFsKGIpO1xuICAgIGV4cGVjdChhLmxlbmd0aCkudG9FcXVhbChiLmxlbmd0aCk7XG4gIH0pO1xuXG4gIGl0KCdNdWx0aXBsZSBhcnJheXMgc2h1ZmZsZSB0b2dldGhlcicsICgpID0+IHtcbiAgICAvLyBDcmVhdGUgMTAwMCBudW1iZXJzIG9yZGVyZWRcbiAgICBjb25zdCBhID0gQXJyYXkuYXBwbHkoMCwge2xlbmd0aDogMTAwMH0pLm1hcChOdW1iZXIuY2FsbCwgTnVtYmVyKS5zbGljZSgxKTtcbiAgICBjb25zdCBiID0gW10uY29uY2F0KGEpOyAgLy8gY29waWVzXG4gICAgY29uc3QgYyA9IFtdLmNvbmNhdChhKTtcbiAgICB1dGlsLnNodWZmbGVDb21ibyhhLCBiKTtcbiAgICBleHBlY3QoYSkubm90LnRvRXF1YWwoYyk7XG4gICAgZXhwZWN0KGEpLnRvRXF1YWwoYik7XG4gICAgZXhwZWN0KGEubGVuZ3RoKS50b0VxdWFsKGMubGVuZ3RoKTtcbiAgfSk7XG5cbiAgaXQoJ0lzIGludGVnZXInLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuaXNJbnQoMC41KSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaXNJbnQoMSkpLnRvQmUodHJ1ZSk7XG4gIH0pO1xuXG4gIGl0KCdTaXplIHRvIHNxdWFyaXNoIHNoYXBlIChwZXJmZWN0IHNxdWFyZSknLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuc2l6ZVRvU3F1YXJpc2hTaGFwZSg5KSkudG9FcXVhbChbMywgM10pO1xuICB9KTtcblxuICBpdCgnU2l6ZSB0byBzcXVhcmlzaCBzaGFwZSAocHJpbWUgbnVtYmVyKScsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5zaXplVG9TcXVhcmlzaFNoYXBlKDExKSkudG9FcXVhbChbNCwgM10pO1xuICB9KTtcblxuICBpdCgnU2l6ZSB0byBzcXVhcmlzaCBzaGFwZSAoYWxtb3N0IHNxdWFyZSknLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuc2l6ZVRvU3F1YXJpc2hTaGFwZSgzNSkpLnRvRXF1YWwoWzYsIDZdKTtcbiAgfSk7XG5cbiAgaXQoJ1NpemUgb2YgMSB0byBzcXVhcmlzaCBzaGFwZScsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5zaXplVG9TcXVhcmlzaFNoYXBlKDEpKS50b0VxdWFsKFsxLCAxXSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSBzaW5nbGUgbnVtYmVyJywgKCkgPT4ge1xuICAgIGV4cGVjdChpbmZlclNoYXBlKDQpKS50b0VxdWFsKFtdKTtcbiAgfSk7XG5cbiAgaXQoJ2luZmVyIHNoYXBlIDFkIGFycmF5JywgKCkgPT4ge1xuICAgIGV4cGVjdChpbmZlclNoYXBlKFsxLCAyLCA1XSkpLnRvRXF1YWwoWzNdKTtcbiAgfSk7XG5cbiAgaXQoJ2luZmVyIHNoYXBlIDJkIGFycmF5JywgKCkgPT4ge1xuICAgIGV4cGVjdChpbmZlclNoYXBlKFtbMSwgMiwgNV0sIFs1LCA0LCAxXV0pKS50b0VxdWFsKFsyLCAzXSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSAzZCBhcnJheScsICgpID0+IHtcbiAgICBjb25zdCBhID0gW1tbMSwgMl0sIFsyLCAzXSwgWzUsIDZdXSwgW1s1LCA2XSwgWzQsIDVdLCBbMSwgMl1dXTtcbiAgICBleHBlY3QoaW5mZXJTaGFwZShhKSkudG9FcXVhbChbMiwgMywgMl0pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgNGQgYXJyYXknLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IFtcbiAgICAgIFtbWzFdLCBbMl1dLCBbWzJdLCBbM11dLCBbWzVdLCBbNl1dXSwgW1tbNV0sIFs2XV0sIFtbNF0sIFs1XV0sIFtbMV0sIFsyXV1dXG4gICAgXTtcbiAgICBleHBlY3QoaW5mZXJTaGFwZShhKSkudG9FcXVhbChbMiwgMywgMiwgMV0pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgb2YgdHlwZWQgYXJyYXknLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDMsIDQsIDVdKTtcbiAgICBleHBlY3QoaW5mZXJTaGFwZShhKSkudG9FcXVhbChbNV0pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgb2YgY2xhbXBlZCB0eXBlZCBhcnJheScsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IFVpbnQ4Q2xhbXBlZEFycmF5KFsxLCAyLCAzLCA0LCA1XSk7XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoYSkpLnRvRXF1YWwoWzVdKTtcbiAgfSk7XG5cbiAgaXQoJ2luZmVyIHNoYXBlIG9mIFVpbnQ4QXJyYXlbXSwgc3RyaW5nIHRlbnNvcicsICgpID0+IHtcbiAgICBjb25zdCBhID0gW25ldyBVaW50OEFycmF5KFsxLCAyXSksIG5ldyBVaW50OEFycmF5KFszLCA0XSldO1xuICAgIGV4cGVjdChpbmZlclNoYXBlKGEsICdzdHJpbmcnKSkudG9FcXVhbChbMl0pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgb2YgVWludDhBcnJheVtdW10sIHN0cmluZyB0ZW5zb3InLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IFtcbiAgICAgIFtuZXcgVWludDhBcnJheShbMV0pLCBuZXcgVWludDhBcnJheShbMl0pXSxcbiAgICAgIFtuZXcgVWludDhBcnJheShbMV0pLCBuZXcgVWludDhBcnJheShbMl0pXVxuICAgIF07XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoYSwgJ3N0cmluZycpKS50b0VxdWFsKFsyLCAyXSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSBvZiBVaW50OEFycmF5W11bXVtdLCBzdHJpbmcgdGVuc29yJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBbXG4gICAgICBbW25ldyBVaW50OEFycmF5KFsxLCAyXSldLCBbbmV3IFVpbnQ4QXJyYXkoWzIsIDFdKV1dLFxuICAgICAgW1tuZXcgVWludDhBcnJheShbMSwgMl0pXSwgW25ldyBVaW50OEFycmF5KFsyLCAxXSldXVxuICAgIF07XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoYSwgJ3N0cmluZycpKS50b0VxdWFsKFsyLCAyLCAxXSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCd1dGlsLmZsYXR0ZW4nLCAoKSA9PiB7XG4gIGl0KCdlbXB0eScsICgpID0+IHtcbiAgICBjb25zdCBkYXRhOiBudW1iZXJbXSA9IFtdO1xuICAgIGV4cGVjdCh1dGlsLmZsYXR0ZW4oZGF0YSkpLnRvRXF1YWwoW10pO1xuICB9KTtcblxuICBpdCgnbmVzdGVkIG51bWJlciBhcnJheXMnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihbWzEsIDIsIDNdLCBbNCwgNSwgNl1dKSkudG9FcXVhbChbMSwgMiwgMywgNCwgNSwgNl0pO1xuICAgIGV4cGVjdCh1dGlsLmZsYXR0ZW4oW1tbMSwgMl0sIFszLCA0XSwgWzUsIDZdLCBbNywgOF1dXSkpLnRvRXF1YWwoW1xuICAgICAgMSwgMiwgMywgNCwgNSwgNiwgNywgOFxuICAgIF0pO1xuICAgIGV4cGVjdCh1dGlsLmZsYXR0ZW4oWzEsIDIsIDMsIDQsIDUsIDZdKSkudG9FcXVhbChbMSwgMiwgMywgNCwgNSwgNl0pO1xuICB9KTtcblxuICBpdCgnbmVzdGVkIHN0cmluZyBhcnJheXMnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihbWydhJywgJ2InXSwgWydjJywgW1snZCddXV1dKSkudG9FcXVhbChbXG4gICAgICAnYScsICdiJywgJ2MnLCAnZCdcbiAgICBdKTtcbiAgICBleHBlY3QodXRpbC5mbGF0dGVuKFtbJ2EnLCBbJ2InXV0sIFsnYycsIFtbJ2QnXV0sICdlJ11dKSkudG9FcXVhbChbXG4gICAgICAnYScsICdiJywgJ2MnLCAnZCcsICdlJ1xuICAgIF0pO1xuICB9KTtcblxuICBpdCgnbWl4ZWQgVHlwZWRBcnJheSBhbmQgbnVtYmVyW10nLCAoKSA9PiB7XG4gICAgY29uc3QgZGF0YSA9XG4gICAgICAgIFtuZXcgRmxvYXQzMkFycmF5KFsxLCAyXSksIDMsIFs0LCA1LCBuZXcgRmxvYXQzMkFycmF5KFs2LCA3XSldXTtcbiAgICBleHBlY3QodXRpbC5mbGF0dGVuKGRhdGEpKS50b0VxdWFsKFsxLCAyLCAzLCA0LCA1LCA2LCA3XSk7XG4gIH0pO1xuXG4gIGl0KCduZXN0ZWQgVWludDhBcnJheXMsIHNraXBUeXBlZEFycmF5PXRydWUnLCAoKSA9PiB7XG4gICAgY29uc3QgZGF0YSA9IFtcbiAgICAgIFtuZXcgVWludDhBcnJheShbMSwgMl0pLCBuZXcgVWludDhBcnJheShbMywgNF0pXSxcbiAgICAgIFtuZXcgVWludDhBcnJheShbNSwgNl0pLCBuZXcgVWludDhBcnJheShbNywgOF0pXVxuICAgIF07XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihkYXRhLCBbXSwgdHJ1ZSkpLnRvRXF1YWwoW1xuICAgICAgbmV3IFVpbnQ4QXJyYXkoWzEsIDJdKSwgbmV3IFVpbnQ4QXJyYXkoWzMsIDRdKSwgbmV3IFVpbnQ4QXJyYXkoWzUsIDZdKSxcbiAgICAgIG5ldyBVaW50OEFycmF5KFs3LCA4XSlcbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ0ludDhBcnJheScsICgpID0+IHtcbiAgICBjb25zdCBkYXRhID0gW25ldyBJbnQ4QXJyYXkoWzEsIDJdKV07XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihkYXRhKSkudG9FcXVhbChbMSwgMl0pO1xuICB9KTtcblxuICBpdCgnaW5kZXggc2lnbmF0dXJlJywgKCkgPT4ge1xuICAgIGNvbnN0IGRhdGE6IHtbaW5kZXg6IG51bWJlcl06IG51bWJlcn0gPSB7MDogMSwgMTogMn07XG4gICAgLy8gV2lsbCBiZSBpZ25vcmVkIHNpbmNlIGFycmF5IGl0ZXJhdGlvbiBpZ25vcmVzIG5lZ2F0aXZlcy5cbiAgICBkYXRhWy0xXSA9IC0xO1xuICAgIC8vIFdpbGwgYmUgaWdub3JlZCBzaW5jZSBub24taW50ZWdlciBhcnJheSBrZXlzIGFyZSBpZ25vcmVkLlxuICAgIGRhdGFbMy4yXSA9IDQ7XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihkYXRhKSkudG9FcXVhbChbMSwgMl0pO1xuICB9KTtcbn0pO1xuXG5mdW5jdGlvbiBlbmNvZGVTdHJpbmdzKGE6IHN0cmluZ1tdKTogVWludDhBcnJheVtdIHtcbiAgcmV0dXJuIGEubWFwKHMgPT4gdXRpbC5lbmNvZGVTdHJpbmcocykpO1xufVxuXG5kZXNjcmliZSgndXRpbC5ieXRlc0Zyb21TdHJpbmdBcnJheScsICgpID0+IHtcbiAgaXQoJ2NvdW50IGJ5dGVzIGFmdGVyIHV0ZjggZW5jb2RpbmcnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuYnl0ZXNGcm9tU3RyaW5nQXJyYXkoZW5jb2RlU3RyaW5ncyhbJ2EnLCAnYmInLCAnY2NjJ10pKSlcbiAgICAgICAgLnRvQmUoNik7XG4gICAgZXhwZWN0KHV0aWwuYnl0ZXNGcm9tU3RyaW5nQXJyYXkoZW5jb2RlU3RyaW5ncyhbJ2EnLCAnYmInLCAnY2NjZGRkJ10pKSlcbiAgICAgICAgLnRvQmUoOSk7XG4gICAgZXhwZWN0KHV0aWwuYnl0ZXNGcm9tU3RyaW5nQXJyYXkoZW5jb2RlU3RyaW5ncyhbJ9C00LDQvdC40LXQuyddKSkpLnRvQmUoNiAqIDIpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5pbmZlckR0eXBlJywgKCkgPT4ge1xuICBpdCgnYSBzaW5nbGUgc3RyaW5nID0+IHN0cmluZycsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKCdoZWxsbycpKS50b0JlKCdzdHJpbmcnKTtcbiAgfSk7XG5cbiAgaXQoJ2Egc2luZ2xlIGJvb2xlYW4gPT4gYm9vbCcsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKHRydWUpKS50b0JlKCdib29sJyk7XG4gICAgZXhwZWN0KHV0aWwuaW5mZXJEdHlwZShmYWxzZSkpLnRvQmUoJ2Jvb2wnKTtcbiAgfSk7XG5cbiAgaXQoJ2Egc2luZ2xlIG51bWJlciA9PiBmbG9hdDMyJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUoMCkpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKDM0KSkudG9CZSgnZmxvYXQzMicpO1xuICB9KTtcblxuICBpdCgnYSBsaXN0IG9mIHN0cmluZ3MgPT4gc3RyaW5nJywgKCkgPT4ge1xuICAgIC8vIEZsYXQuXG4gICAgZXhwZWN0KHV0aWwuaW5mZXJEdHlwZShbJ2EnLCAnYicsICdjJ10pKS50b0JlKCdzdHJpbmcnKTtcbiAgICAvLyBOZXN0ZWQuXG4gICAgZXhwZWN0KHV0aWwuaW5mZXJEdHlwZShbXG4gICAgICBbWydhJ11dLCBbWydiJ11dLCBbWydjJ11dLCBbWydkJ11dXG4gICAgXSkpLnRvQmUoJ3N0cmluZycpO1xuICB9KTtcblxuICBpdCgnYSBsaXN0IG9mIGJvb2xzID0+IGZsb2F0MzInLCAoKSA9PiB7XG4gICAgLy8gRmxhdC5cbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKFtmYWxzZSwgdHJ1ZSwgZmFsc2VdKSkudG9CZSgnYm9vbCcpO1xuICAgIC8vIE5lc3RlZC5cbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKFtcbiAgICAgIFtbdHJ1ZV1dLCBbW2ZhbHNlXV0sIFtbdHJ1ZV1dLCBbW3RydWVdXVxuICAgIF0pKS50b0JlKCdib29sJyk7XG4gIH0pO1xuXG4gIGl0KCdhIGxpc3Qgb2YgbnVtYmVycyA9PiBmbG9hdDMyJywgKCkgPT4ge1xuICAgIC8vIEZsYXQuXG4gICAgZXhwZWN0KHV0aWwuaW5mZXJEdHlwZShbMCwgMSwgMl0pKS50b0JlKCdmbG9hdDMyJyk7XG4gICAgLy8gTmVzdGVkLlxuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUoW1tbMF1dLCBbWzFdXSwgW1syXV0sIFtbM11dXSkpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwucmVwZWF0ZWRUcnknLCAoKSA9PiB7XG4gIGl0KCdyZXNvbHZlcycsIChkb25lRm4pID0+IHtcbiAgICBsZXQgY291bnRlciA9IDA7XG4gICAgY29uc3QgY2hlY2tGbiA9ICgpID0+IHtcbiAgICAgIGNvdW50ZXIrKztcbiAgICAgIGlmIChjb3VudGVyID09PSAyKSB7XG4gICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgfVxuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH07XG5cbiAgICB1dGlsLnJlcGVhdGVkVHJ5KGNoZWNrRm4pLnRoZW4oZG9uZUZuKS5jYXRjaCgoKSA9PiB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ1JlamVjdGVkIGJhY2tvZmYuJyk7XG4gICAgfSk7XG4gIH0pO1xuICBpdCgncmVqZWN0cycsIChkb25lRm4pID0+IHtcbiAgICBjb25zdCBjaGVja0ZuID0gKCkgPT4gZmFsc2U7XG5cbiAgICB1dGlsLnJlcGVhdGVkVHJ5KGNoZWNrRm4sICgpID0+IDAsIDUpXG4gICAgICAgIC50aGVuKCgpID0+IHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0JhY2tvZmYgcmVzb2x2ZWQnKTtcbiAgICAgICAgfSlcbiAgICAgICAgLmNhdGNoKGRvbmVGbik7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCd1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUnLCAoKSA9PiB7XG4gIGl0KCdlbXB0eSBzaGFwZScsICgpID0+IHtcbiAgICBjb25zdCByZXN1bHQgPSB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoW10sIDApO1xuICAgIGV4cGVjdChyZXN1bHQpLnRvRXF1YWwoW10pO1xuICB9KTtcblxuICBpdCgnWzIsIDMsIDRdIC0+IFsyLCAzLCA0XScsICgpID0+IHtcbiAgICBjb25zdCByZXN1bHQgPSB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoWzIsIDMsIDRdLCAyNCk7XG4gICAgZXhwZWN0KHJlc3VsdCkudG9FcXVhbChbMiwgMywgNF0pO1xuICB9KTtcblxuICBpdCgnWzIsIC0xLCA0XSAtPiBbMiwgMywgNF0sIHNpemU9MjQnLCAoKSA9PiB7XG4gICAgY29uc3QgcmVzdWx0ID0gdXRpbC5pbmZlckZyb21JbXBsaWNpdFNoYXBlKFsyLCAtMSwgNF0sIDI0KTtcbiAgICBleHBlY3QocmVzdWx0KS50b0VxdWFsKFsyLCAzLCA0XSk7XG4gIH0pO1xuXG4gIGl0KCdbLTEsIDMsIDRdIC0+IFsyLCAzLCA0XSwgc2l6ZT0yNCcsICgpID0+IHtcbiAgICBjb25zdCByZXN1bHQgPSB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoWy0xLCAzLCA0XSwgMjQpO1xuICAgIGV4cGVjdChyZXN1bHQpLnRvRXF1YWwoWzIsIDMsIDRdKTtcbiAgfSk7XG5cbiAgaXQoJ1syLCAzLCAtMV0gLT4gWzIsIDMsIDRdLCBzaXplPTI0JywgKCkgPT4ge1xuICAgIGNvbnN0IHJlc3VsdCA9IHV0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZShbMiwgMywgLTFdLCAyNCk7XG4gICAgZXhwZWN0KHJlc3VsdCkudG9FcXVhbChbMiwgMywgNF0pO1xuICB9KTtcblxuICBpdCgnWzIsIC0xLCAtMV0gdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoWzIsIC0xLCAtMV0sIDI0KSkudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCdbMiwgMywgLTFdIHNpemU9MTMgdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoWzIsIDMsIC0xXSwgMTMpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ1syLCAzLCA0XSBzaXplPTI1IChzaG91bGQgYmUgMjQpIHRocm93cyBlcnJvcicsICgpID0+IHtcbiAgICBleHBlY3QoKCkgPT4gdXRpbC5pbmZlckZyb21JbXBsaWNpdFNoYXBlKFsyLCAzLCA0XSwgMjUpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwgcGFyc2VBeGlzUGFyYW0nLCAoKSA9PiB7XG4gIGl0KCdheGlzPW51bGwgcmV0dXJucyBubyBheGVzIGZvciBzY2FsYXInLCAoKSA9PiB7XG4gICAgY29uc3QgYXhpczogbnVtYmVyID0gbnVsbDtcbiAgICBjb25zdCBzaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvRXF1YWwoW10pO1xuICB9KTtcblxuICBpdCgnYXhpcz1udWxsIHJldHVybnMgMCBheGlzIGZvciBUZW5zb3IxRCcsICgpID0+IHtcbiAgICBjb25zdCBheGlzOiBudW1iZXIgPSBudWxsO1xuICAgIGNvbnN0IHNoYXBlID0gWzRdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbMF0pO1xuICB9KTtcblxuICBpdCgnYXhpcz1udWxsIHJldHVybnMgYWxsIGF4ZXMgZm9yIFRlbnNvcjNEJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXM6IG51bWJlcltdID0gbnVsbDtcbiAgICBjb25zdCBzaGFwZSA9IFszLCAxLCAyXTtcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvRXF1YWwoWzAsIDEsIDJdKTtcbiAgfSk7XG5cbiAgaXQoJ2F4aXMgYXMgYSBzaW5nbGUgbnVtYmVyJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSAxO1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDEsIDJdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbMV0pO1xuICB9KTtcblxuICBpdCgnYXhpcyBhcyBzaW5nbGUgbmVnYXRpdmUgbnVtYmVyJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSAtMTtcbiAgICBjb25zdCBzaGFwZSA9IFszLCAxLCAyXTtcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvRXF1YWwoWzJdKTtcblxuICAgIGNvbnN0IGF4aXMyID0gLTI7XG4gICAgZXhwZWN0KHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpczIsIHNoYXBlKSkudG9FcXVhbChbMV0pO1xuXG4gICAgY29uc3QgYXhpczMgPSAtMztcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzMywgc2hhcGUpKS50b0VxdWFsKFswXSk7XG4gIH0pO1xuXG4gIGl0KCdheGlzIGFzIGxpc3Qgb2YgbmVnYXRpdmUgbnVtYmVycycsICgpID0+IHtcbiAgICBjb25zdCBheGlzID0gWy0xLCAtM107XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgMSwgMl07XG4gICAgZXhwZWN0KHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b0VxdWFsKFsyLCAwXSk7XG4gIH0pO1xuXG4gIGl0KCdheGlzIGFzIGxpc3Qgb2YgcG9zaXRpdmUgbnVtYmVycycsICgpID0+IHtcbiAgICBjb25zdCBheGlzID0gWzAsIDJdO1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDEsIDJdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbMCwgMl0pO1xuICB9KTtcblxuICBpdCgnYXhpcyBhcyBjb21ibyBvZiBwb3NpdGl2ZSBhbmQgbmVnYXRpdmUgbnVtYmVycycsICgpID0+IHtcbiAgICBjb25zdCBheGlzID0gWzAsIC0xXTtcbiAgICBjb25zdCBzaGFwZSA9IFszLCAxLCAyXTtcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvRXF1YWwoWzAsIDJdKTtcbiAgfSk7XG5cbiAgaXQoJ2F4aXMgb3V0IG9mIHJhbmdlIHRocm93cyBlcnJvcicsICgpID0+IHtcbiAgICBjb25zdCBheGlzID0gLTQ7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgMSwgMl07XG4gICAgZXhwZWN0KCgpID0+IHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b1Rocm93RXJyb3IoKTtcblxuICAgIGNvbnN0IGF4aXMyID0gNDtcbiAgICBleHBlY3QoKCkgPT4gdXRpbC5wYXJzZUF4aXNQYXJhbShheGlzMiwgc2hhcGUpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ2F4aXMgYSBsaXN0IHdpdGggb25lIG51bWJlciBvdXQgb2YgcmFuZ2UgdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSBbMCwgNF07XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgMSwgMl07XG4gICAgZXhwZWN0KCgpID0+IHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ2F4aXMgd2l0aCBkZWNpbWFsIHZhbHVlIHRocm93cyBlcnJvcicsICgpID0+IHtcbiAgICBjb25zdCBheGlzID0gMC41O1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDEsIDJdO1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCd1dGlsLnNxdWVlemVTaGFwZScsICgpID0+IHtcbiAgaXQoJ3NjYWxhcicsICgpID0+IHtcbiAgICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKFtdKTtcbiAgICBleHBlY3QobmV3U2hhcGUpLnRvRXF1YWwoW10pO1xuICAgIGV4cGVjdChrZXB0RGltcykudG9FcXVhbChbXSk7XG4gIH0pO1xuXG4gIGl0KCcxeDEgcmVkdWNlZCB0byBzY2FsYXInLCAoKSA9PiB7XG4gICAgY29uc3Qge25ld1NoYXBlLCBrZXB0RGltc30gPSB1dGlsLnNxdWVlemVTaGFwZShbMSwgMV0pO1xuICAgIGV4cGVjdChuZXdTaGFwZSkudG9FcXVhbChbXSk7XG4gICAgZXhwZWN0KGtlcHREaW1zKS50b0VxdWFsKFtdKTtcbiAgfSk7XG5cbiAgaXQoJzF4M3gxIHJlZHVjZWQgdG8gWzNdJywgKCkgPT4ge1xuICAgIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoWzEsIDMsIDFdKTtcbiAgICBleHBlY3QobmV3U2hhcGUpLnRvRXF1YWwoWzNdKTtcbiAgICBleHBlY3Qoa2VwdERpbXMpLnRvRXF1YWwoWzFdKTtcbiAgfSk7XG5cbiAgaXQoJzF4MXg0IHJlZHVjZWQgdG8gWzRdJywgKCkgPT4ge1xuICAgIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoWzEsIDEsIDRdKTtcbiAgICBleHBlY3QobmV3U2hhcGUpLnRvRXF1YWwoWzRdKTtcbiAgICBleHBlY3Qoa2VwdERpbXMpLnRvRXF1YWwoWzJdKTtcbiAgfSk7XG5cbiAgaXQoJzJ4M3g0IG5vdCByZWR1Y3Rpb24nLCAoKSA9PiB7XG4gICAgY29uc3Qge25ld1NoYXBlLCBrZXB0RGltc30gPSB1dGlsLnNxdWVlemVTaGFwZShbMiwgMywgNF0pO1xuICAgIGV4cGVjdChuZXdTaGFwZSkudG9FcXVhbChbMiwgMywgNF0pO1xuICAgIGV4cGVjdChrZXB0RGltcykudG9FcXVhbChbMCwgMSwgMl0pO1xuICB9KTtcblxuICBkZXNjcmliZSgnd2l0aCBheGlzJywgKCkgPT4ge1xuICAgIGl0KCdzaG91bGQgb25seSByZWR1Y2UgZGltZW5zaW9ucyBzcGVjaWZpZWQgYnkgYXhpcycsICgpID0+IHtcbiAgICAgIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoWzEsIDEsIDEsIDEsIDRdLCBbMSwgMl0pO1xuICAgICAgZXhwZWN0KG5ld1NoYXBlKS50b0VxdWFsKFsxLCAxLCA0XSk7XG4gICAgICBleHBlY3Qoa2VwdERpbXMpLnRvRXF1YWwoWzAsIDMsIDRdKTtcbiAgICB9KTtcbiAgICBpdCgnc2hvdWxkIG9ubHkgcmVkdWNlIGRpbWVuc2lvbnMgc3BlY2lmaWVkIGJ5IG5lZ2F0aXZlIGF4aXMnLCAoKSA9PiB7XG4gICAgICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxLCAxLCAxLCA0XSwgWy0yLCAtM10pO1xuICAgICAgZXhwZWN0KG5ld1NoYXBlKS50b0VxdWFsKFsxLCAxLCA0XSk7XG4gICAgICBleHBlY3Qoa2VwdERpbXMpLnRvRXF1YWwoWzAsIDEsIDRdKTtcbiAgICB9KTtcbiAgICBpdCgnc2hvdWxkIG9ubHkgcmVkdWNlIGRpbWVuc2lvbnMgc3BlY2lmaWVkIGJ5IG5lZ2F0aXZlIGF4aXMnLCAoKSA9PiB7XG4gICAgICBjb25zdCBheGlzID0gWy0yLCAtM107XG4gICAgICB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgMSwgMSwgNF0sIGF4aXMpO1xuICAgICAgZXhwZWN0KGF4aXMpLnRvRXF1YWwoWy0yLCAtM10pO1xuICAgIH0pO1xuICAgIGl0KCd0aHJvd3MgZXJyb3Igd2hlbiBzcGVjaWZpZWQgYXhpcyBpcyBub3Qgc3F1ZWV6YWJsZScsICgpID0+IHtcbiAgICAgIGV4cGVjdCgoKSA9PiB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgMiwgMSwgNF0sIFsxLCAyXSkpLnRvVGhyb3dFcnJvcigpO1xuICAgIH0pO1xuICAgIGl0KCd0aHJvd3MgZXJyb3Igd2hlbiBzcGVjaWZpZWQgbmVnYXRpdmUgYXhpcyBpcyBub3Qgc3F1ZWV6YWJsZScsICgpID0+IHtcbiAgICAgIGV4cGVjdCgoKSA9PiB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgMiwgMSwgNF0sIFstMSwgLTJdKSkudG9UaHJvd0Vycm9yKCk7XG4gICAgfSk7XG4gICAgaXQoJ3Rocm93cyBlcnJvciB3aGVuIHNwZWNpZmllZCBheGlzIGlzIG91dCBvZiByYW5nZScsICgpID0+IHtcbiAgICAgIGV4cGVjdCgoKSA9PiB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgMiwgMSwgNF0sIFsxMSwgMjJdKSkudG9UaHJvd0Vycm9yKCk7XG4gICAgfSk7XG4gICAgaXQoJ3Rocm93cyBlcnJvciB3aGVuIHNwZWNpZmllZCBuZWdhdGl2ZSBheGlzIGlzIG91dCBvZiByYW5nZScsICgpID0+IHtcbiAgICAgIGV4cGVjdCgoKSA9PiB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgMiwgMSwgNF0sIFtcbiAgICAgICAgLTExLCAtMjJcbiAgICAgIF0pKS50b1Rocm93RXJyb3IoKTtcbiAgICB9KTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwuY2hlY2tDb252ZXJzaW9uRm9yRXJyb3JzJywgKCkgPT4ge1xuICBpdCgnRmxvYXQzMkFycmF5IGhhcyBOYU4nLCAoKSA9PiB7XG4gICAgZXhwZWN0KFxuICAgICAgICAoKSA9PiB1dGlsLmNoZWNrQ29udmVyc2lvbkZvckVycm9ycyhcbiAgICAgICAgICAgIG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDMsIE5hTiwgNCwgMjU1XSksICdmbG9hdDMyJykpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ0Zsb2F0MzJBcnJheSBoYXMgSW5maW5pdHknLCAoKSA9PiB7XG4gICAgZXhwZWN0KFxuICAgICAgICAoKSA9PiB1dGlsLmNoZWNrQ29udmVyc2lvbkZvckVycm9ycyhcbiAgICAgICAgICAgIG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDMsIEluZmluaXR5LCA0LCAyNTVdKSwgJ2Zsb2F0MzInKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgnSW50MzJBcnJheSBoYXMgTmFOJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLmNoZWNrQ29udmVyc2lvbkZvckVycm9ycyhbMSwgMiwgMywgNCwgTmFOXSwgJ2ludDMyJykpXG4gICAgICAgIC50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwuaGFzRW5jb2RpbmdMb3NzJywgKCkgPT4ge1xuICBpdCgnY29tcGxleDY0IHRvIGFueScsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2NvbXBsZXg2NCcsICdjb21wbGV4NjQnKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdjb21wbGV4NjQnLCAnZmxvYXQzMicpKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnY29tcGxleDY0JywgJ2ludDMyJykpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdjb21wbGV4NjQnLCAnYm9vbCcpKS50b0JlKHRydWUpO1xuICB9KTtcblxuICBpdCgnYW55IHRvIGNvbXBsZXg2NCcsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2Jvb2wnLCAnY29tcGxleDY0JykpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnaW50MzInLCAnY29tcGxleDY0JykpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnZmxvYXQzMicsICdjb21wbGV4NjQnKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdjb21wbGV4NjQnLCAnY29tcGxleDY0JykpLnRvQmUoZmFsc2UpO1xuICB9KTtcblxuICBpdCgnYW55IHRvIGZsb2F0MzInLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdib29sJywgJ2Zsb2F0MzInKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdpbnQzMicsICdmbG9hdDMyJykpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnZmxvYXQzMicsICdmbG9hdDMyJykpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnY29tcGxleDY0JywgJ2Zsb2F0MzInKSkudG9CZSh0cnVlKTtcbiAgfSk7XG5cbiAgaXQoJ2Zsb2F0MzIgdG8gYW55JywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnZmxvYXQzMicsICdmbG9hdDMyJykpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnZmxvYXQzMicsICdpbnQzMicpKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnZmxvYXQzMicsICdib29sJykpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdmbG9hdDMyJywgJ2NvbXBsZXg2NCcpKS50b0JlKGZhbHNlKTtcbiAgfSk7XG5cbiAgaXQoJ2ludDMyIHRvIGxvd2VyJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnaW50MzInLCAnaW50MzInKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdpbnQzMicsICdib29sJykpLnRvQmUodHJ1ZSk7XG4gIH0pO1xuXG4gIGl0KCdsb3dlciB0byBpbnQzMicsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2Jvb2wnLCAnaW50MzInKSkudG9CZShmYWxzZSk7XG4gIH0pO1xuXG4gIGl0KCdib29sIHRvIGJvb2wnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdib29sJywgJ2Jvb2wnKSkudG9CZShmYWxzZSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCd1dGlsLnRvTmVzdGVkQXJyYXknLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnMiBkaW1lbnNpb25zJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLCAyLCAzLCA0LCA1LCA2XSk7XG4gICAgZXhwZWN0KHV0aWwudG9OZXN0ZWRBcnJheShbMiwgM10sIGEpKS50b0VxdWFsKFtbMSwgMiwgM10sIFs0LCA1LCA2XV0pO1xuICB9KTtcblxuICBpdCgnMyBkaW1lbnNpb25zICgyeDJ4MyknLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoWzAsIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMV0pO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoWzIsIDIsIDNdLCBhKSkudG9FcXVhbChbXG4gICAgICBbWzAsIDEsIDJdLCBbMywgNCwgNV1dLCBbWzYsIDcsIDhdLCBbOSwgMTAsIDExXV1cbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJzMgZGltZW5zaW9ucyAoM3gyeDIpJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgRmxvYXQzMkFycmF5KFswLCAxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMCwgMTFdKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KFszLCAyLCAyXSwgYSkpLnRvRXF1YWwoW1xuICAgICAgW1swLCAxXSwgWzIsIDNdXSwgW1s0LCA1XSwgWzYsIDddXSwgW1s4LCA5XSwgWzEwLCAxMV1dXG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCdpbnZhbGlkIGRpbWVuc2lvbicsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgM10pO1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLnRvTmVzdGVkQXJyYXkoWzIsIDJdLCBhKSkudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCd0ZW5zb3IgdG8gbmVzdGVkIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHggPSB0ZW5zb3IyZChbMSwgMiwgMywgNF0sIFsyLCAyXSk7XG4gICAgZXhwZWN0KHV0aWwudG9OZXN0ZWRBcnJheSh4LnNoYXBlLCBhd2FpdCB4LmRhdGEoKSkpLnRvRXF1YWwoW1xuICAgICAgWzEsIDJdLCBbMywgNF1cbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ3NjYWxhciB0byBuZXN0ZWQgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgeCA9IHNjYWxhcigxKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KHguc2hhcGUsIGF3YWl0IHguZGF0YSgpKSkudG9FcXVhbCgxKTtcbiAgfSk7XG5cbiAgaXQoJ3RlbnNvciB3aXRoIHplcm8gc2hhcGUnLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoWzAsIDFdKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KFsxLCAwLCAyXSwgYSkpLnRvRXF1YWwoW10pO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZVdpdGhGbGFncygndXRpbC50b05lc3RlZEFycmF5IGZvciBhIGNvbXBsZXggdGVuc29yJywgQUxMX0VOVlMsICgpID0+IHtcbiAgaXQoJzIgZGltZW5zaW9ucycsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMTEsIDIsIDEyLCAzLCAxMywgNCwgMTQsIDUsIDE1LCA2LCAxNl0pO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoWzIsIDNdLCBhLCB0cnVlKSkudG9FcXVhbChbXG4gICAgICBbMSwgMTEsIDIsIDEyLCAzLCAxM10sIFs0LCAxNCwgNSwgMTUsIDYsIDE2XVxuICAgIF0pO1xuICB9KTtcblxuICBpdCgnMyBkaW1lbnNpb25zICgyeDJ4MyknLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoW1xuICAgICAgMCwgNTAsIDEsIDUxLCAyLCA1MiwgMywgNTMsIDQsICA1NCwgNSwgIDU1LFxuICAgICAgNiwgNTYsIDcsIDU3LCA4LCA1OCwgOSwgNTksIDEwLCA2MCwgMTEsIDYxXG4gICAgXSk7XG4gICAgZXhwZWN0KHV0aWwudG9OZXN0ZWRBcnJheShbMiwgMiwgM10sIGEsIHRydWUpKS50b0VxdWFsKFtcbiAgICAgIFtbMCwgNTAsIDEsIDUxLCAyLCA1Ml0sIFszLCA1MywgNCwgNTQsIDUsIDU1XV0sXG4gICAgICBbWzYsIDU2LCA3LCA1NywgOCwgNThdLCBbOSwgNTksIDEwLCA2MCwgMTEsIDYxXV1cbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJzMgZGltZW5zaW9ucyAoM3gyeDIpJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgRmxvYXQzMkFycmF5KFtcbiAgICAgIDAsIDUwLCAxLCA1MSwgMiwgNTIsIDMsIDUzLCA0LCAgNTQsIDUsICA1NSxcbiAgICAgIDYsIDU2LCA3LCA1NywgOCwgNTgsIDksIDU5LCAxMCwgNjAsIDExLCA2MVxuICAgIF0pO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoWzMsIDIsIDJdLCBhLCB0cnVlKSkudG9FcXVhbChbXG4gICAgICBbWzAsIDUwLCAxLCA1MV0sIFsyLCA1MiwgMywgNTNdXSwgW1s0LCA1NCwgNSwgNTVdLCBbNiwgNTYsIDcsIDU3XV0sXG4gICAgICBbWzgsIDU4LCA5LCA1OV0sIFsxMCwgNjAsIDExLCA2MV1dXG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCdpbnZhbGlkIGRpbWVuc2lvbicsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMTEsIDIsIDEyLCAzLCAxM10pO1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLnRvTmVzdGVkQXJyYXkoWzIsIDJdLCBhLCB0cnVlKSkudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCd0ZW5zb3IgdG8gbmVzdGVkIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHggPSBjb21wbGV4KFtbMSwgMl0sIFszLCA0XV0sIFtbMTEsIDEyXSwgWzEzLCAxNF1dKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KHguc2hhcGUsIGF3YWl0IHguZGF0YSgpLCB0cnVlKSkudG9FcXVhbChbXG4gICAgICBbMSwgMTEsIDIsIDEyXSwgWzMsIDEzLCA0LCAxNF1cbiAgICBdKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwuZmV0Y2gnLCAoKSA9PiB7XG4gIGl0KCdzaG91bGQgY2FsbCB0aGUgcGxhdGZvcm0gZmV0Y2gnLCAoKSA9PiB7XG4gICAgc3B5T24odGYuZW52KCkucGxhdGZvcm0sICdmZXRjaCcpXG4gICAgICAgIC5hbmQuY2FsbEZha2UoYXN5bmMgKCkgPT4gKHt9IGFzIHVua25vd24gYXMgUmVzcG9uc2UpKTtcblxuICAgIHV0aWwuZmV0Y2goJ3Rlc3QvcGF0aCcsIHttZXRob2Q6ICdHRVQnfSk7XG5cbiAgICBleHBlY3QodGYuZW52KCkucGxhdGZvcm0uZmV0Y2gpLnRvSGF2ZUJlZW5DYWxsZWRXaXRoKCd0ZXN0L3BhdGgnLCB7XG4gICAgICBtZXRob2Q6ICdHRVQnXG4gICAgfSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCd1dGlsLmVuY29kZVN0cmluZycsICgpID0+IHtcbiAgaXQoJ0VuY29kZSBhbiBlbXB0eSBzdHJpbmcsIGRlZmF1bHQgZW5jb2RpbmcnLCAoKSA9PiB7XG4gICAgY29uc3QgcmVzID0gdXRpbC5lbmNvZGVTdHJpbmcoJycpO1xuICAgIGV4cGVjdChyZXMpLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkoW10pKTtcbiAgfSk7XG5cbiAgaXQoJ0VuY29kZSBhbiBlbXB0eSBzdHJpbmcsIHV0Zi04IGVuY29kaW5nJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlcyA9IHV0aWwuZW5jb2RlU3RyaW5nKCcnLCAndXRmLTgnKTtcbiAgICBleHBlY3QocmVzKS50b0VxdWFsKG5ldyBVaW50OEFycmF5KFtdKSk7XG4gIH0pO1xuXG4gIGl0KCdFbmNvZGUgYW4gZW1wdHkgc3RyaW5nLCBpbnZhbGlkIGRlY29kaW5nJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLmVuY29kZVN0cmluZygnJywgJ2Zvb2JhcmJheCcpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ0VuY29kZSBjeXJpbGxpYyBsZXR0ZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlcyA9IHV0aWwuZW5jb2RlU3RyaW5nKCdLYdC6byDRgdGCZScpO1xuICAgIGV4cGVjdChyZXMpLnRvRXF1YWwoXG4gICAgICAgIG5ldyBVaW50OEFycmF5KFs3NSwgOTcsIDIwOCwgMTg2LCAxMTEsIDMyLCAyMDksIDEyOSwgMjA5LCAxMzAsIDEwMV0pKTtcbiAgfSk7XG5cbiAgaXQoJ0VuY29kZSBhc2NpaSBsZXR0ZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlcyA9IHV0aWwuZW5jb2RlU3RyaW5nKCdoZWxsbycpO1xuICAgIGV4cGVjdChyZXMpLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkoWzEwNCwgMTAxLCAxMDgsIDEwOCwgMTExXSkpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5kZWNvZGVTdHJpbmcnLCAoKSA9PiB7XG4gIGl0KCdkZWNvZGUgYW4gZW1wdHkgc3RyaW5nJywgKCkgPT4ge1xuICAgIGNvbnN0IHMgPSB1dGlsLmRlY29kZVN0cmluZyhuZXcgVWludDhBcnJheShbXSkpO1xuICAgIGV4cGVjdChzKS50b0VxdWFsKCcnKTtcbiAgfSk7XG5cbiAgaXQoJ2RlY29kZSBhc2NpaScsICgpID0+IHtcbiAgICBjb25zdCBzID0gdXRpbC5kZWNvZGVTdHJpbmcobmV3IFVpbnQ4QXJyYXkoWzEwNCwgMTAxLCAxMDgsIDEwOCwgMTExXSkpO1xuICAgIGV4cGVjdChzKS50b0VxdWFsKCdoZWxsbycpO1xuICB9KTtcblxuICBpdCgnZGVjb2RlIGN5cmlsbGljJywgKCkgPT4ge1xuICAgIGNvbnN0IHMgPSB1dGlsLmRlY29kZVN0cmluZyhcbiAgICAgICAgbmV3IFVpbnQ4QXJyYXkoWzc1LCA5NywgMjA4LCAxODYsIDExMSwgMzIsIDIwOSwgMTI5LCAyMDksIDEzMCwgMTAxXSkpO1xuICAgIGV4cGVjdChzKS50b0VxdWFsKCdLYdC6byDRgdGCZScpO1xuICB9KTtcblxuICBpdCgnZGVjb2RlIHV0Zi0xNicsICgpID0+IHtcbiAgICBjb25zdCBzID0gdXRpbC5kZWNvZGVTdHJpbmcoXG4gICAgICAgIG5ldyBVaW50OEFycmF5KFsyNTUsIDI1NCwgMjM3LCAxMzksIDAsIDEzOCwgNCwgODksIDYsIDExNl0pLCAndXRmLTE2Jyk7XG5cbiAgICAvLyBVVEYtMTYgYWxsb3dzIG9wdGlvbmFsIHByZXNlbmNlIG9mIGJ5dGUtb3JkZXItbWFyayAoQk9NKVxuICAgIC8vIENvbnN0cnVjdCBzdHJpbmcgZm9yICfor63oqIDlpITnkIYnLCB3aXRoIGFuZCB3aXRob3V0IEJPTVxuICAgIGNvbnN0IGV4cGVjdGVkID0gU3RyaW5nLmZyb21Db2RlUG9pbnQoMHg4YmVkLCAweDhhMDAsIDB4NTkwNCwgMHg3NDA2KTtcbiAgICBjb25zdCBleHBlY3RlZEJPTSA9XG4gICAgICAgIFN0cmluZy5mcm9tQ29kZVBvaW50KDB4ZmVmZiwgMHg4YmVkLCAweDhhMDAsIDB4NTkwNCwgMHg3NDA2KTtcblxuICAgIGlmIChzLmNvZGVQb2ludEF0KDApID09PSAweGZlZmYpIHtcbiAgICAgIGV4cGVjdChzKS50b0VxdWFsKGV4cGVjdGVkQk9NKTtcbiAgICB9IGVsc2Uge1xuICAgICAgZXhwZWN0KHMpLnRvRXF1YWwoZXhwZWN0ZWQpO1xuICAgIH1cbiAgfSk7XG5cbiAgaXQoJ2Fzc2VydCBwcm9taXNlJywgKCkgPT4ge1xuICAgIGNvbnN0IHByb21pc2UgPSBuZXcgUHJvbWlzZSgoKSA9PiB7fSk7XG4gICAgZXhwZWN0KHV0aWwuaXNQcm9taXNlKHByb21pc2UpKS50b0JlVHJ1dGh5KCk7XG4gICAgY29uc3QgcHJvbWlzZTIgPSB7dGhlbjogKCkgPT4ge319O1xuICAgIGV4cGVjdCh1dGlsLmlzUHJvbWlzZShwcm9taXNlMikpLnRvQmVUcnV0aHkoKTtcbiAgICBjb25zdCBwcm9taXNlMyA9IHt9O1xuICAgIGV4cGVjdCh1dGlsLmlzUHJvbWlzZShwcm9taXNlMykpLnRvQmVGYWxzeSgpO1xuICB9KTtcbn0pO1xuIl19