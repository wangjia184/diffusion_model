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
import { ENGINE } from './engine';
import { inferShape } from './tensor_util_env';
import { arraysEqual, encodeString, flatten, isString, isTypedArray } from './util';
const TEST_EPSILON_FLOAT32 = 1e-3;
export const TEST_EPSILON_FLOAT16 = 1e-1;
export function expectArraysClose(actual, expected, epsilon) {
    if (epsilon == null) {
        epsilon = testEpsilon();
    }
    return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, epsilon));
}
export function testEpsilon() {
    return ENGINE.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
        TEST_EPSILON_FLOAT16;
}
function expectArraysPredicate(actual, expected, predicate) {
    let checkClassType = true;
    if (isTypedArray(actual) || isTypedArray(expected)) {
        checkClassType = false;
    }
    if (isTypedArray(actual) && isTypedArray(expected)) {
        checkClassType = true;
    }
    if (checkClassType) {
        const aType = actual.constructor.name;
        const bType = expected.constructor.name;
        if (aType !== bType) {
            throw new Error(`Arrays are of different type. Actual: ${aType}. ` +
                `Expected: ${bType}`);
        }
    }
    if (Array.isArray(actual) && Array.isArray(expected)) {
        const actualShape = inferShape(actual);
        const expectedShape = inferShape(expected);
        if (!arraysEqual(actualShape, expectedShape)) {
            throw new Error(`Arrays have different shapes. ` +
                `Actual: [${actualShape}]. Expected: [${expectedShape}]`);
        }
    }
    const actualFlat = isTypedArray(actual) ? actual : flatten(actual);
    const expectedFlat = isTypedArray(expected) ?
        expected :
        flatten(expected);
    if (actualFlat.length !== expectedFlat.length) {
        throw new Error(`Arrays have different lengths actual: ${actualFlat.length} vs ` +
            `expected: ${expectedFlat.length}.\n` +
            `Actual:   ${actualFlat}.\n` +
            `Expected: ${expectedFlat}.`);
    }
    for (let i = 0; i < expectedFlat.length; ++i) {
        const a = actualFlat[i];
        const e = expectedFlat[i];
        if (!predicate(a, e)) {
            throw new Error(`Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
                `Actual:   ${actualFlat}.\n` +
                `Expected: ${expectedFlat}.`);
        }
    }
    if (typeof expect !== 'undefined') {
        expect().nothing();
    }
}
export function expectPromiseToFail(fn, done) {
    fn().then(() => done.fail(), () => done());
    if (typeof expect !== 'undefined') {
        expect().nothing();
    }
}
export function expectArraysEqual(actual, expected) {
    const exp = typeof expected === 'string' || typeof expected === 'number' ||
        typeof expected === 'boolean' ?
        [expected] :
        expected;
    if (isString(actual) || isString(actual[0]) ||
        isString(expected) || isString(expected[0])) {
        // tslint:disable-next-line: triple-equals
        return expectArraysPredicate(actual, exp, (a, b) => a == b);
    }
    return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, 0));
}
export function expectNumbersClose(a, e, epsilon) {
    if (epsilon == null) {
        epsilon = testEpsilon();
    }
    if (!areClose(a, e, epsilon)) {
        throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
    }
    if (typeof expect !== 'undefined') {
        expect().nothing();
    }
}
function areClose(a, e, epsilon) {
    if (!isFinite(a) && !isFinite(e)) {
        return true;
    }
    if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
        return false;
    }
    return true;
}
export function expectValuesInRange(actual, low, high) {
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] < low || actual[i] > high) {
            throw new Error(`Value out of range:${actual[i]} low: ${low}, high: ${high}`);
        }
    }
}
export function expectArrayBuffersEqual(actual, expected) {
    // Safari does not like comparing ArrayBuffers directly. Wrapping in
    // a Float32Array solves this issue.
    const actualArray = new Float32Array(actual);
    const expectedArray = new Float32Array(expected);
    if (actualArray.length !== expectedArray.length) {
        throw new Error('Expected ArrayBuffer to be of length ' +
            `${expectedArray.length}, but it was ${actualArray.length}`);
    }
    for (let i = 0; i < expectedArray.length; i++) {
        if (actualArray[i] !== expectedArray[i]) {
            throw new Error(`Expected ArrayBuffer value at ${i} to be ` +
                `${expectedArray[i]} but got ${actualArray[i]} instead`);
        }
    }
}
/** Encodes strings into utf-8 bytes. */
export function encodeStrings(a) {
    for (let i = 0; i < a.length; i++) {
        const val = a[i];
        if (Array.isArray(val)) {
            encodeStrings(val);
        }
        else {
            a[i] = encodeString(val);
        }
    }
    return a;
}
/** Creates an HTMLVideoElement with autoplay-friendly default settings. */
export function createVideoElement(source) {
    const video = document.createElement('video');
    if ('playsInline' in video) {
        // tslint:disable-next-line:no-any
        video.playsInline = true;
    }
    video.muted = true;
    video.loop = true;
    video.style.position = 'fixed';
    video.style.left = '0px';
    video.style.top = '0px';
    video.preload = 'auto';
    video.appendChild(source);
    return new Promise(resolve => {
        video.addEventListener('loadeddata', _ => resolve(video));
        video.load();
    });
}
export async function play(video) {
    await video.play();
    if ('requestVideoFrameCallback' in video) {
        await new Promise(resolve => {
            // tslint:disable-next-line:no-any
            video.requestVideoFrameCallback(resolve);
        });
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVzdF91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy90ZXN0X3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUNoQyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFFN0MsT0FBTyxFQUFDLFdBQVcsRUFBRSxZQUFZLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxZQUFZLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFbEYsTUFBTSxvQkFBb0IsR0FBRyxJQUFJLENBQUM7QUFDbEMsTUFBTSxDQUFDLE1BQU0sb0JBQW9CLEdBQUcsSUFBSSxDQUFDO0FBRXpDLE1BQU0sVUFBVSxpQkFBaUIsQ0FDN0IsTUFBZ0QsRUFDaEQsUUFBa0QsRUFBRSxPQUFnQjtJQUN0RSxJQUFJLE9BQU8sSUFBSSxJQUFJLEVBQUU7UUFDbkIsT0FBTyxHQUFHLFdBQVcsRUFBRSxDQUFDO0tBQ3pCO0lBQ0QsT0FBTyxxQkFBcUIsQ0FDeEIsTUFBTSxFQUFFLFFBQVEsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFXLEVBQUUsQ0FBVyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7QUFDL0UsQ0FBQztBQUVELE1BQU0sVUFBVSxXQUFXO0lBQ3pCLE9BQU8sTUFBTSxDQUFDLE9BQU8sQ0FBQyxjQUFjLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLG9CQUFvQixDQUFDLENBQUM7UUFDdEIsb0JBQW9CLENBQUM7QUFDdkUsQ0FBQztBQUVELFNBQVMscUJBQXFCLENBQzFCLE1BQWtCLEVBQUUsUUFBb0IsRUFDeEMsU0FBb0M7SUFDdEMsSUFBSSxjQUFjLEdBQUcsSUFBSSxDQUFDO0lBQzFCLElBQUksWUFBWSxDQUFDLE1BQU0s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QUFJLENBQUM7S0FDbkM7SUFDRCxLQUFLLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQztJQUNuQixLQUFLLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztJQUNsQixLQUFLLENBQUMsS0FBSyxDQUFDLFFBQVEsR0FBRyxPQUFPLENBQUM7SUFDL0IsS0FBSyxDQUFDLEtBQUssQ0FBQyxJQUFJLEdBQUcsS0FBSyxDQUFDO0lBQ3pCLEtBQUssQ0FBQyxLQUFLLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQztJQUV4QixLQUFLLENBQUMsT0FBTyxHQUFHLE1BQU0sQ0FBQztJQUN2QixLQUFLLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQzFCLE9BQU8sSUFBSSxPQUFPLENBQUMsT0FBTyxDQUFDLEVBQUU7UUFDM0IsS0FBSyxDQUFDLGdCQUFnQixDQUFDLFlBQVksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1FBQzFELEtBQUssQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUNmLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQztBQUVELE1BQU0sQ0FBQyxLQUFLLFVBQVUsSUFBSSxDQUFDLEtBQXVCO0lBQ2hELE1BQU0sS0FBSyxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ25CLElBQUksMkJBQTJCLElBQUksS0FBSyxFQUFFO1FBQ3hDLE1BQU0sSUFBSSxPQUFPLENBQUMsT0FBTyxDQUFDLEVBQUU7WUFDMUIsa0NBQWtDO1lBQ2pDLEtBQWEsQ0FBQyx5QkFBeUIsQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNwRCxDQUFDLENBQUMsQ0FBQztLQUNKO0FBQ0gsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4vZW5naW5lJztcbmltcG9ydCB7aW5mZXJTaGFwZX0gZnJvbSAnLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtSZWN1cnNpdmVBcnJheSwgVGVuc29yTGlrZSwgVHlwZWRBcnJheX0gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQge2FycmF5c0VxdWFsLCBlbmNvZGVTdHJpbmcsIGZsYXR0ZW4sIGlzU3RyaW5nLCBpc1R5cGVkQXJyYXl9IGZyb20gJy4vdXRpbCc7XG5cbmNvbnN0IFRFU1RfRVBTSUxPTl9GTE9BVDMyID0gMWUtMztcbmV4cG9ydCBjb25zdCBURVNUX0VQU0lMT05fRkxPQVQxNiA9IDFlLTE7XG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RBcnJheXNDbG9zZShcbiAgICBhY3R1YWw6IFR5cGVkQXJyYXl8bnVtYmVyfFJlY3Vyc2l2ZUFycmF5PG51bWJlcj4sXG4gICAgZXhwZWN0ZWQ6IFR5cGVkQXJyYXl8bnVtYmVyfFJlY3Vyc2l2ZUFycmF5PG51bWJlcj4sIGVwc2lsb24/OiBudW1iZXIpIHtcbiAgaWYgKGVwc2lsb24gPT0gbnVsbCkge1xuICAgIGVwc2lsb24gPSB0ZXN0RXBzaWxvbigpO1xuICB9XG4gIHJldHVybiBleHBlY3RBcnJheXNQcmVkaWNhdGUoXG4gICAgICBhY3R1YWwsIGV4cGVjdGVkLCAoYSwgYikgPT4gYXJlQ2xvc2UoYSBhcyBudW1iZXIsIGIgYXMgbnVtYmVyLCBlcHNpbG9uKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB0ZXN0RXBzaWxvbigpIHtcbiAgcmV0dXJuIEVOR0lORS5iYWNrZW5kLmZsb2F0UHJlY2lzaW9uKCkgPT09IDMyID8gVEVTVF9FUFNJTE9OX0ZMT0FUMzIgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBURVNUX0VQU0lMT05fRkxPQVQxNjtcbn1cblxuZnVuY3Rpb24gZXhwZWN0QXJyYXlzUHJlZGljYXRlKFxuICAgIGFjdHVhbDogVGVuc29yTGlrZSwgZXhwZWN0ZWQ6IFRlbnNvckxpa2UsXG4gICAgcHJlZGljYXRlOiAoYToge30sIGI6IHt9KSA9PiBib29sZWFuKSB7XG4gIGxldCBjaGVja0NsYXNzVHlwZSA9IHRydWU7XG4gIGlmIChpc1R5cGVkQXJyYXkoYWN0dWFsKSB8fCBpc1R5cGVkQXJyYXkoZXhwZWN0ZWQpKSB7XG4gICAgY2hlY2tDbGFzc1R5cGUgPSBmYWxzZTtcbiAgfVxuICBpZiAoaXNUeXBlZEFycmF5KGFjdHVhbCkgJiYgaXNUeXBlZEFycmF5KGV4cGVjdGVkKSkge1xuICAgIGNoZWNrQ2xhc3NUeXBlID0gdHJ1ZTtcbiAgfVxuICBpZiAoY2hlY2tDbGFzc1R5cGUpIHtcbiAgICBjb25zdCBhVHlwZSA9IGFjdHVhbC5jb25zdHJ1Y3Rvci5uYW1lO1xuICAgIGNvbnN0IGJUeXBlID0gZXhwZWN0ZWQuY29uc3RydWN0b3IubmFtZTtcblxuICAgIGlmIChhVHlwZSAhPT0gYlR5cGUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQXJyYXlzIGFyZSBvZiBkaWZmZXJlbnQgdHlwZS4gQWN0dWFsOiAke2FUeXBlfS4gYCArXG4gICAgICAgICAgYEV4cGVjdGVkOiAke2JUeXBlfWApO1xuICAgIH1cbiAgfVxuXG4gIGlmIChBcnJheS5pc0FycmF5KGFjdHVhbCkgJiYgQXJyYXkuaXNBcnJheShleHBlY3RlZCkpIHtcbiAgICBjb25zdCBhY3R1YWxTaGFwZSA9IGluZmVyU2hhcGUoYWN0dWFsKTtcbiAgICBjb25zdCBleHBlY3RlZFNoYXBlID0gaW5mZXJTaGFwZShleHBlY3RlZCk7XG4gICAgaWYgKCFhcnJheXNFcXVhbChhY3R1YWxTaGFwZSwgZXhwZWN0ZWRTaGFwZSkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQXJyYXlzIGhhdmUgZGlmZmVyZW50IHNoYXBlcy4gYCArXG4gICAgICAgICAgYEFjdHVhbDogWyR7YWN0dWFsU2hhcGV9XS4gRXhwZWN0ZWQ6IFske2V4cGVjdGVkU2hhcGV9XWApO1xuICAgIH1cbiAgfVxuXG4gIGNvbnN0IGFjdHVhbEZsYXQgPVxuICAgICAgaXNUeXBlZEFycmF5KGFjdHVhbCkgPyBhY3R1YWwgOiBmbGF0dGVuKGFjdHVhbCBhcyBSZWN1cnNpdmVBcnJheTxudW1iZXI+KTtcbiAgY29uc3QgZXhwZWN0ZWRGbGF0ID0gaXNUeXBlZEFycmF5KGV4cGVjdGVkKSA/XG4gICAgICBleHBlY3RlZCA6XG4gICAgICBmbGF0dGVuKGV4cGVjdGVkIGFzIFJlY3Vyc2l2ZUFycmF5PG51bWJlcj4pO1xuXG4gIGlmIChhY3R1YWxGbGF0Lmxlbmd0aCAhPT0gZXhwZWN0ZWRGbGF0Lmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYEFycmF5cyBoYXZlIGRpZmZlcmVudCBsZW5ndGhzIGFjdHVhbDogJHthY3R1YWxGbGF0Lmxlbmd0aH0gdnMgYCArXG4gICAgICAgIGBleHBlY3RlZDogJHtleHBlY3RlZEZsYXQubGVuZ3RofS5cXG5gICtcbiAgICAgICAgYEFjdHVhbDogICAke2FjdHVhbEZsYXR9LlxcbmAgK1xuICAgICAgICBgRXhwZWN0ZWQ6ICR7ZXhwZWN0ZWRGbGF0fS5gKTtcbiAgfVxuICBmb3IgKGxldCBpID0gMDsgaSA8IGV4cGVjdGVkRmxhdC5sZW5ndGg7ICsraSkge1xuICAgIGNvbnN0IGEgPSBhY3R1YWxGbGF0W2ldO1xuICAgIGNvbnN0IGUgPSBleHBlY3RlZEZsYXRbaV07XG5cbiAgICBpZiAoIXByZWRpY2F0ZShhLCBlKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBBcnJheXMgZGlmZmVyOiBhY3R1YWxbJHtpfV0gPSAke2F9LCBleHBlY3RlZFske2l9XSA9ICR7ZX0uXFxuYCArXG4gICAgICAgICAgYEFjdHVhbDogICAke2FjdHVhbEZsYXR9LlxcbmAgK1xuICAgICAgICAgIGBFeHBlY3RlZDogJHtleHBlY3RlZEZsYXR9LmApO1xuICAgIH1cbiAgfVxuICBpZiAodHlwZW9mIGV4cGVjdCAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICBleHBlY3QoKS5ub3RoaW5nKCk7XG4gIH1cbn1cblxuZXhwb3J0IGludGVyZmFjZSBEb25lRm4ge1xuICAoKTogdm9pZDtcbiAgZmFpbDogKG1lc3NhZ2U/OiBFcnJvcnxzdHJpbmcpID0+IHZvaWQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RQcm9taXNlVG9GYWlsKGZuOiAoKSA9PiBQcm9taXNlPHt9PiwgZG9uZTogRG9uZUZuKTogdm9pZCB7XG4gIGZuKCkudGhlbigoKSA9PiBkb25lLmZhaWwoKSwgKCkgPT4gZG9uZSgpKTtcbiAgaWYgKHR5cGVvZiBleHBlY3QgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgZXhwZWN0KCkubm90aGluZygpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RBcnJheXNFcXVhbChhY3R1YWw6IFRlbnNvckxpa2UsIGV4cGVjdGVkOiBUZW5zb3JMaWtlKSB7XG4gIGNvbnN0IGV4cCA9IHR5cGVvZiBleHBlY3RlZCA9PT0gJ3N0cmluZycgfHwgdHlwZW9mIGV4cGVjdGVkID09PSAnbnVtYmVyJyB8fFxuICAgICAgICAgIHR5cGVvZiBleHBlY3RlZCA9PT0gJ2Jvb2xlYW4nID9cbiAgICAgIFtleHBlY3RlZF0gYXMgbnVtYmVyW10gOlxuICAgICAgZXhwZWN0ZWQgYXMgbnVtYmVyW107XG4gIGlmIChpc1N0cmluZyhhY3R1YWwpIHx8IGlzU3RyaW5nKChhY3R1YWwgYXMgc3RyaW5nW10pWzBdKSB8fFxuICAgICAgaXNTdHJpbmcoZXhwZWN0ZWQpIHx8IGlzU3RyaW5nKChleHBlY3RlZCBhcyBzdHJpbmdbXSlbMF0pKSB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiB0cmlwbGUtZXF1YWxzXG4gICAgcmV0dXJuIGV4cGVjdEFycmF5c1ByZWRpY2F0ZShhY3R1YWwsIGV4cCwgKGEsIGIpID0+IGEgPT0gYik7XG4gIH1cbiAgcmV0dXJuIGV4cGVjdEFycmF5c1ByZWRpY2F0ZShcbiAgICAgIGFjdHVhbCwgZXhwZWN0ZWQsIChhLCBiKSA9PiBhcmVDbG9zZShhIGFzIG51bWJlciwgYiBhcyBudW1iZXIsIDApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGV4cGVjdE51bWJlcnNDbG9zZShhOiBudW1iZXIsIGU6IG51bWJlciwgZXBzaWxvbj86IG51bWJlcikge1xuICBpZiAoZXBzaWxvbiA9PSBudWxsKSB7XG4gICAgZXBzaWxvbiA9IHRlc3RFcHNpbG9uKCk7XG4gIH1cbiAgaWYgKCFhcmVDbG9zZShhLCBlLCBlcHNpbG9uKSkge1xuICAgIHRocm93IG5ldyBFcnJvcihgTnVtYmVycyBkaWZmZXI6IGFjdHVhbCA9PT0gJHthfSwgZXhwZWN0ZWQgPT09ICR7ZX1gKTtcbiAgfVxuICBpZiAodHlwZW9mIGV4cGVjdCAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICBleHBlY3QoKS5ub3RoaW5nKCk7XG4gIH1cbn1cblxuZnVuY3Rpb24gYXJlQ2xvc2UoYTogbnVtYmVyLCBlOiBudW1iZXIsIGVwc2lsb246IG51bWJlcik6IGJvb2xlYW4ge1xuICBpZiAoIWlzRmluaXRlKGEpICYmICFpc0Zpbml0ZShlKSkge1xuICAgIHJldHVybiB0cnVlO1xuICB9XG4gIGlmIChpc05hTihhKSB8fCBpc05hTihlKSB8fCBNYXRoLmFicyhhIC0gZSkgPiBlcHNpbG9uKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZXhwZWN0VmFsdWVzSW5SYW5nZShcbiAgICBhY3R1YWw6IFR5cGVkQXJyYXl8bnVtYmVyW10sIGxvdzogbnVtYmVyLCBoaWdoOiBudW1iZXIpIHtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhY3R1YWwubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoYWN0dWFsW2ldIDwgbG93IHx8IGFjdHVhbFtpXSA+IGhpZ2gpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgVmFsdWUgb3V0IG9mIHJhbmdlOiR7YWN0dWFsW2ldfSBsb3c6ICR7bG93fSwgaGlnaDogJHtoaWdofWApO1xuICAgIH1cbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZXhwZWN0QXJyYXlCdWZmZXJzRXF1YWwoXG4gICAgYWN0dWFsOiBBcnJheUJ1ZmZlciwgZXhwZWN0ZWQ6IEFycmF5QnVmZmVyKSB7XG4gIC8vIFNhZmFyaSBkb2VzIG5vdCBsaWtlIGNvbXBhcmluZyBBcnJheUJ1ZmZlcnMgZGlyZWN0bHkuIFdyYXBwaW5nIGluXG4gIC8vIGEgRmxvYXQzMkFycmF5IHNvbHZlcyB0aGlzIGlzc3VlLlxuICBjb25zdCBhY3R1YWxBcnJheSA9IG5ldyBGbG9hdDMyQXJyYXkoYWN0dWFsKTtcbiAgY29uc3QgZXhwZWN0ZWRBcnJheSA9IG5ldyBGbG9hdDMyQXJyYXkoZXhwZWN0ZWQpO1xuICBpZiAoYWN0dWFsQXJyYXkubGVuZ3RoICE9PSBleHBlY3RlZEFycmF5Lmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ0V4cGVjdGVkIEFycmF5QnVmZmVyIHRvIGJlIG9mIGxlbmd0aCAnICtcbiAgICAgICAgYCR7ZXhwZWN0ZWRBcnJheS5sZW5ndGh9LCBidXQgaXQgd2FzICR7YWN0dWFsQXJyYXkubGVuZ3RofWApO1xuICB9XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBleHBlY3RlZEFycmF5Lmxlbmd0aDsgaSsrKSB7XG4gICAgaWYgKGFjdHVhbEFycmF5W2ldICE9PSBleHBlY3RlZEFycmF5W2ldKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYEV4cGVjdGVkIEFycmF5QnVmZmVyIHZhbHVlIGF0ICR7aX0gdG8gYmUgYCArXG4gICAgICAgICAgYCR7ZXhwZWN0ZWRBcnJheVtpXX0gYnV0IGdvdCAke2FjdHVhbEFycmF5W2ldfSBpbnN0ZWFkYCk7XG4gICAgfVxuICB9XG59XG5cbi8qKiBFbmNvZGVzIHN0cmluZ3MgaW50byB1dGYtOCBieXRlcy4gKi9cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGVTdHJpbmdzKGE6IFJlY3Vyc2l2ZUFycmF5PHt9Pik6XG4gICAgUmVjdXJzaXZlQXJyYXk8VWludDhBcnJheT4ge1xuICBmb3IgKGxldCBpID0gMDsgaSA8IChhIGFzIEFycmF5PHt9PikubGVuZ3RoOyBpKyspIHtcbiAgICBjb25zdCB2YWwgPSBhW2ldO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHZhbCkpIHtcbiAgICAgIGVuY29kZVN0cmluZ3ModmFsKTtcbiAgICB9IGVsc2Uge1xuICAgICAgYVtpXSA9IGVuY29kZVN0cmluZyh2YWwgYXMgc3RyaW5nKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIGEgYXMgUmVjdXJzaXZlQXJyYXk8VWludDhBcnJheT47XG59XG5cbi8qKiBDcmVhdGVzIGFuIEhUTUxWaWRlb0VsZW1lbnQgd2l0aCBhdXRvcGxheS1mcmllbmRseSBkZWZhdWx0IHNldHRpbmdzLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVZpZGVvRWxlbWVudChzb3VyY2U6IEhUTUxTb3VyY2VFbGVtZW50KTpcbiAgICBQcm9taXNlPEhUTUxWaWRlb0VsZW1lbnQ+IHtcbiAgY29uc3QgdmlkZW8gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCd2aWRlbycpO1xuICBpZiAoJ3BsYXlzSW5saW5lJyBpbiB2aWRlbykge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAodmlkZW8gYXMgYW55KS5wbGF5c0lubGluZSA9IHRydWU7XG4gIH1cbiAgdmlkZW8ubXV0ZWQgPSB0cnVlO1xuICB2aWRlby5sb29wID0gdHJ1ZTtcbiAgdmlkZW8uc3R5bGUucG9zaXRpb24gPSAnZml4ZWQnO1xuICB2aWRlby5zdHlsZS5sZWZ0ID0gJzBweCc7XG4gIHZpZGVvLnN0eWxlLnRvcCA9ICcwcHgnO1xuXG4gIHZpZGVvLnByZWxvYWQgPSAnYXV0byc7XG4gIHZpZGVvLmFwcGVuZENoaWxkKHNvdXJjZSk7XG4gIHJldHVybiBuZXcgUHJvbWlzZShyZXNvbHZlID0+IHtcbiAgICB2aWRlby5hZGRFdmVudExpc3RlbmVyKCdsb2FkZWRkYXRhJywgXyA9PiByZXNvbHZlKHZpZGVvKSk7XG4gICAgdmlkZW8ubG9hZCgpO1xuICB9KTtcbn1cblxuZXhwb3J0IGFzeW5jIGZ1bmN0aW9uIHBsYXkodmlkZW86IEhUTUxWaWRlb0VsZW1lbnQpIHtcbiAgYXdhaXQgdmlkZW8ucGxheSgpO1xuICBpZiAoJ3JlcXVlc3RWaWRlb0ZyYW1lQ2FsbGJhY2snIGluIHZpZGVvKSB7XG4gICAgYXdhaXQgbmV3IFByb21pc2UocmVzb2x2ZSA9PiB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAodmlkZW8gYXMgYW55KS5yZXF1ZXN0VmlkZW9GcmFtZUNhbGxiYWNrKHJlc29sdmUpO1xuICAgIH0pO1xuICB9XG59XG4iXX0=