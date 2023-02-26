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
import { env } from './environment';
import * as base from './util_base';
export * from './util_base';
export * from './hash_util';
/**
 * Create typed array for scalar value. Used for storing in `DataStorage`.
 */
export function createScalarValue(value, dtype) {
    if (dtype === 'string') {
        return encodeString(value);
    }
    return toTypedArray([value], dtype);
}
function noConversionNeeded(a, dtype) {
    return (a instanceof Float32Array && dtype === 'float32') ||
        (a instanceof Int32Array && dtype === 'int32') ||
        (a instanceof Uint8Array && dtype === 'bool');
}
export function toTypedArray(a, dtype) {
    if (dtype === 'string') {
        throw new Error('Cannot convert a string[] to a TypedArray');
    }
    if (Array.isArray(a)) {
        a = flatten(a);
    }
    if (env().getBool('DEBUG')) {
        base.checkConversionForErrors(a, dtype);
    }
    if (noConversionNeeded(a, dtype)) {
        return a;
    }
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(a);
    }
    else if (dtype === 'int32') {
        return new Int32Array(a);
    }
    else if (dtype === 'bool') {
        const bool = new Uint8Array(a.length);
        for (let i = 0; i < bool.length; ++i) {
            if (Math.round(a[i]) !== 0) {
                bool[i] = 1;
            }
        }
        return bool;
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
/**
 * Returns the current high-resolution time in milliseconds relative to an
 * arbitrary time in the past. It works across different platforms (node.js,
 * browsers).
 *
 * ```js
 * console.log(tf.util.now());
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function now() {
    return env().platform.now();
}
/**
 * Returns a platform-specific implementation of
 * [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 *
 * If `fetch` is defined on the global object (`window`, `process`, etc.),
 * `tf.util.fetch` returns that function.
 *
 * If not, `tf.util.fetch` returns a platform-specific solution.
 *
 * ```js
 * const resource = await tf.util.fetch('https://unpkg.com/@tensorflow/tfjs');
 * // handle response
 * ```
 *
 * @doc {heading: 'Util'}
 */
export function fetch(path, requestInits) {
    return env().platform.fetch(path, requestInits);
}
/**
 * Encodes the provided string into bytes using the provided encoding scheme.
 *
 * @param s The string to encode.
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
export function encodeString(s, encoding = 'utf-8') {
    encoding = encoding || 'utf-8';
    return env().platform.encode(s, encoding);
}
/**
 * Decodes the provided bytes into a string using the provided encoding scheme.
 * @param bytes The bytes to decode.
 *
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
export function decodeString(bytes, encoding = 'utf-8') {
    encoding = encoding || 'utf-8';
    return env().platform.decode(bytes, encoding);
}
export function isTypedArray(a) {
    return env().platform.isTypedArray(a);
}
// NOTE: We explicitly type out what T extends instead of any so that
// util.flatten on a nested array of number doesn't try to infer T as a
// number[][], causing us to explicitly type util.flatten<number>().
/**
 *  Flattens an arbitrarily nested array.
 *
 * ```js
 * const a = [[1, 2], [3, 4], [5, [6, [7]]]];
 * const flat = tf.util.flatten(a);
 * console.log(flat);
 * ```
 *
 *  @param arr The nested array to flatten.
 *  @param result The destination array which holds the elements.
 *  @param skipTypedArray If true, avoids flattening the typed arrays. Defaults
 *      to false.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function flatten(arr, result = [], skipTypedArray = false) {
    if (result == null) {
        result = [];
    }
    if (typeof arr === 'boolean' || typeof arr === 'number' ||
        typeof arr === 'string' || base.isPromise(arr) || arr == null ||
        isTypedArray(arr) && skipTypedArray) {
        result.push(arr);
    }
    else if (Array.isArray(arr) || isTypedArray(arr)) {
        for (let i = 0; i < arr.length; ++i) {
            flatten(arr[i], result, skipTypedArray);
        }
    }
    else {
        let maxIndex = -1;
        for (const key of Object.keys(arr)) {
            // 0 or positive integer.
            if (/^([1-9]+[0-9]*|0)$/.test(key)) {
                maxIndex = Math.max(maxIndex, Number(key));
            }
        }
        for (let i = 0; i <= maxIndex; i++) {
            // tslint:disable-next-line: no-unnecessary-type-assertion
            flatten(arr[i], result, skipTypedArray);
        }
    }
    return result;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvdXRpbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxDLE9BQU8sS0FBSyxJQUFJLE1BQU0sYUFBYSxDQUFDO0FBQ3BDLGNBQWMsYUFBYSxDQUFDO0FBQzVCLGNBQWMsYUFBYSxDQUFDO0FBRTVCOztHQUVHO0FBQ0gsTUFBTSxVQUFVLGlCQUFpQixDQUM3QixLQUFlLEVBQUUsS0FBZTtJQUNsQyxJQUFJLEtBQUssS0FBSyxRQUFRLEVBQUU7UUFDdEIsT0FBTyxZQUFZLENBQUMsS0FBSyxDQUFDLENBQUM7S0FDNUI7SUFFRCxPQUFPLFlBQVksQ0FBQyxDQUFDLEtBQUssQ0FBQyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQ3RDLENBQUM7QUFFRCxTQUFTLGtCQUFrQixDQUFDLENBQWEsRUFBRSxLQUFlO0lBQ3hELE9BQU8sQ0FBQyxDQUFDLFlBQVksWUFBWSxJQUFJLEtBQUssS0FBSyxTQUFTLENBQUM7UUFDckQsQ0FBQyxDQUFDLFlBQVksVUFBVSxJQUFJLEtBQUssS0FBSyxPQUFPLENBQUM7UUFDOUMsQ0FBQyxDQUFDLFlBQVksVUFBVSxJQUFJLEtBQUssS0FBSyxNQUFNLENBQUMsQ0FBQztBQUNwRCxDQUFDO0FBRUQsTUFBTSxVQUFVLFlBQVksQ0FBQyxDQUFhLEVBQUUsS0FBZTtJQUN6RCxJQUFJLEtBQUssS0FBSyxRQUFRLEVBQUU7UUFDdEIsTUFBTSxJQUFJLEtBQUssQ0FBQywyQ0FBMkMsQ0FBQyxDQUFDO0tBQzlEO0lBQ0QsSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFO1FBQ3BCLENBQUMsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDaEI7SUFFRCxJQUFJLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsRUFBRTtRQUMxQixJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBYSxFQUFFLEtBQUssQ0FBQyxDQUFDO0tBQ3JEO0lBQ0QsSUFBSSxrQkFBa0IsQ0FBQyxDQUFDLEVBQUUsS0FBSyxDQUFDLEVBQUU7UUFDaEMsT0FBTyxDQUFlLENBQUM7S0FDeEI7SUFDRCxJQUFJLEtBQUssSUFBSSxJQUFJLElBQUksS0FBSyxLQUFLLFNBQVMsSUFBSSxLQUFLLEtBQUssV0FBVyxFQUFFO1FBQ2pFLE9BQU8sSUFBSSxZQUFZLENBQUMsQ0FBYSxDQUFDLENBQUM7S0FDeEM7U0FBTSxJQUFJLEtBQUssS0FBSyxPQUFPLEVBQUU7UUFDNUIsT0FBTyxJQUFJLFVBQVUsQ0FBQyxDQUFhLENBQUMsQ0FBQztLQUN0QztTQUFNLElBQUksS0FBSyxLQUFLLE1BQU0sRUFBRTtRQUMzQixNQUFNLElBQUksR0FBRyxJQUFJLFVBQVUsQ0FBRSxDQUFjLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDcEQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDcEMsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFFLENBQWMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFBRTtnQkFDeEMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQzthQUNiO1NBQ0Y7UUFDRCxPQUFPLElBQUksQ0FBQztLQUNiO1NBQU07UUFDTCxNQUFNLElBQUksS0FBSyxDQUFDLHFCQUFxQixLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQy9DO0FBQ0gsQ0FBQztBQUVEOzs7Ozs7Ozs7O0dBVUc7QUFDSCxNQUFNLFVBQVUsR0FBRztJQUNqQixPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxHQUFHLEVBQUUsQ0FBQztBQUM5QixDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsTUFBTSxVQUFVLEtBQUssQ0FDakIsSUFBWSxFQUFFLFlBQTBCO0lBQzFDLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsWUFBWSxDQUFDLENBQUM7QUFDbEQsQ0FBQztBQUVEOzs7Ozs7O0dBT0c7QUFDSCxNQUFNLFVBQVUsWUFBWSxDQUFDLENBQVMsRUFBRSxRQUFRLEdBQUcsT0FBTztJQUN4RCxRQUFRLEdBQUcsUUFBUSxJQUFJLE9BQU8sQ0FBQztJQUMvQixPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxDQUFDO0FBQzVDLENBQUM7QUFFRDs7Ozs7OztHQU9HO0FBQ0gsTUFBTSxVQUFVLFlBQVksQ0FBQyxLQUFpQixFQUFFLFFBQVEsR0FBRyxPQUFPO0lBQ2hFLFFBQVEsR0FBRyxRQUFRLElBQUksT0FBTyxDQUFDO0lBQy9CLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLENBQUM7QUFDaEQsQ0FBQztBQUVELE1BQU0sVUFBVSxZQUFZLENBQUMsQ0FBSztJQUVoQyxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELHFFQUFxRTtBQUNyRSx1RUFBdUU7QUFDdkUsb0VBQW9FO0FBQ3BFOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE1BQU0sVUFDTixPQUFPLENBQ0gsR0FBd0IsRUFBRSxTQUFjLEVBQUUsRUFBRSxjQUFjLEdBQUcsS0FBSztJQUNwRSxJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7UUFDbEIsTUFBTSxHQUFHLEVBQUUsQ0FBQztLQUNiO0lBQ0QsSUFBSSxPQUFPLEdBQUcsS0FBSyxTQUFTLElBQUksT0FBTyxHQUFHLEtBQUssUUFBUTtRQUNyRCxPQUFPLEdBQUcsS0FBSyxRQUFRLElBQUksSUFBSSxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsSUFBSSxHQUFHLElBQUksSUFBSTtRQUMzRCxZQUFZLENBQUMsR0FBRyxDQUFDLElBQUksY0FBYyxFQUFFO1FBQ3ZDLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBUSxDQUFDLENBQUM7S0FDdkI7U0FBTSxJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLElBQUksWUFBWSxDQUFDLEdBQUcsQ0FBQyxFQUFFO1FBQ2xELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxHQUFHLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1lBQ25DLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsTUFBTSxFQUFFLGNBQWMsQ0FBQyxDQUFDO1NBQ3pDO0tBQ0Y7U0FBTTtRQUNMLElBQUksUUFBUSxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBQ2xCLEtBQUssTUFBTSxHQUFHLElBQUksTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRTtZQUNsQyx5QkFBeUI7WUFDekIsSUFBSSxvQkFBb0IsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUU7Z0JBQ2xDLFFBQVEsR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLFFBQVEsRUFBRSxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQzthQUM1QztTQUNGO1FBQ0QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxJQUFJLFFBQVEsRUFBRSxDQUFDLEVBQUUsRUFBRTtZQUNsQywwREFBMEQ7WUFDMUQsT0FBTyxDQUFFLEdBQXlCLENBQUMsQ0FBQyxDQUFDLEVBQUUsTUFBTSxFQUFFLGNBQWMsQ0FBQyxDQUFDO1NBQ2hFO0tBQ0Y7SUFDRCxPQUFPLE1BQU0sQ0FBQztBQUNoQixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2Vudn0gZnJvbSAnLi9lbnZpcm9ubWVudCc7XG5pbXBvcnQge0JhY2tlbmRWYWx1ZXMsIERhdGFUeXBlLCBSZWN1cnNpdmVBcnJheSwgVGVuc29yTGlrZSwgVHlwZWRBcnJheX0gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQgKiBhcyBiYXNlIGZyb20gJy4vdXRpbF9iYXNlJztcbmV4cG9ydCAqIGZyb20gJy4vdXRpbF9iYXNlJztcbmV4cG9ydCAqIGZyb20gJy4vaGFzaF91dGlsJztcblxuLyoqXG4gKiBDcmVhdGUgdHlwZWQgYXJyYXkgZm9yIHNjYWxhciB2YWx1ZS4gVXNlZCBmb3Igc3RvcmluZyBpbiBgRGF0YVN0b3JhZ2VgLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU2NhbGFyVmFsdWUoXG4gICAgdmFsdWU6IERhdGFUeXBlLCBkdHlwZTogRGF0YVR5cGUpOiBCYWNrZW5kVmFsdWVzIHtcbiAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgIHJldHVybiBlbmNvZGVTdHJpbmcodmFsdWUpO1xuICB9XG5cbiAgcmV0dXJuIHRvVHlwZWRBcnJheShbdmFsdWVdLCBkdHlwZSk7XG59XG5cbmZ1bmN0aW9uIG5vQ29udmVyc2lvbk5lZWRlZChhOiBUZW5zb3JMaWtlLCBkdHlwZTogRGF0YVR5cGUpOiBib29sZWFuIHtcbiAgcmV0dXJuIChhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5ICYmIGR0eXBlID09PSAnZmxvYXQzMicpIHx8XG4gICAgICAoYSBpbnN0YW5jZW9mIEludDMyQXJyYXkgJiYgZHR5cGUgPT09ICdpbnQzMicpIHx8XG4gICAgICAoYSBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkgJiYgZHR5cGUgPT09ICdib29sJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB0b1R5cGVkQXJyYXkoYTogVGVuc29yTGlrZSwgZHR5cGU6IERhdGFUeXBlKTogVHlwZWRBcnJheSB7XG4gIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0Nhbm5vdCBjb252ZXJ0IGEgc3RyaW5nW10gdG8gYSBUeXBlZEFycmF5Jyk7XG4gIH1cbiAgaWYgKEFycmF5LmlzQXJyYXkoYSkpIHtcbiAgICBhID0gZmxhdHRlbihhKTtcbiAgfVxuXG4gIGlmIChlbnYoKS5nZXRCb29sKCdERUJVRycpKSB7XG4gICAgYmFzZS5jaGVja0NvbnZlcnNpb25Gb3JFcnJvcnMoYSBhcyBudW1iZXJbXSwgZHR5cGUpO1xuICB9XG4gIGlmIChub0NvbnZlcnNpb25OZWVkZWQoYSwgZHR5cGUpKSB7XG4gICAgcmV0dXJuIGEgYXMgVHlwZWRBcnJheTtcbiAgfVxuICBpZiAoZHR5cGUgPT0gbnVsbCB8fCBkdHlwZSA9PT0gJ2Zsb2F0MzInIHx8IGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgIHJldHVybiBuZXcgRmxvYXQzMkFycmF5KGEgYXMgbnVtYmVyW10pO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnaW50MzInKSB7XG4gICAgcmV0dXJuIG5ldyBJbnQzMkFycmF5KGEgYXMgbnVtYmVyW10pO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnYm9vbCcpIHtcbiAgICBjb25zdCBib29sID0gbmV3IFVpbnQ4QXJyYXkoKGEgYXMgbnVtYmVyW10pLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBib29sLmxlbmd0aDsgKytpKSB7XG4gICAgICBpZiAoTWF0aC5yb3VuZCgoYSBhcyBudW1iZXJbXSlbaV0pICE9PSAwKSB7XG4gICAgICAgIGJvb2xbaV0gPSAxO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gYm9vbDtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZGF0YSB0eXBlICR7ZHR5cGV9YCk7XG4gIH1cbn1cblxuLyoqXG4gKiBSZXR1cm5zIHRoZSBjdXJyZW50IGhpZ2gtcmVzb2x1dGlvbiB0aW1lIGluIG1pbGxpc2Vjb25kcyByZWxhdGl2ZSB0byBhblxuICogYXJiaXRyYXJ5IHRpbWUgaW4gdGhlIHBhc3QuIEl0IHdvcmtzIGFjcm9zcyBkaWZmZXJlbnQgcGxhdGZvcm1zIChub2RlLmpzLFxuICogYnJvd3NlcnMpLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zb2xlLmxvZyh0Zi51dGlsLm5vdygpKTtcbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdVdGlsJywgbmFtZXNwYWNlOiAndXRpbCd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBub3coKTogbnVtYmVyIHtcbiAgcmV0dXJuIGVudigpLnBsYXRmb3JtLm5vdygpO1xufVxuXG4vKipcbiAqIFJldHVybnMgYSBwbGF0Zm9ybS1zcGVjaWZpYyBpbXBsZW1lbnRhdGlvbiBvZlxuICogW2BmZXRjaGBdKGh0dHBzOi8vZGV2ZWxvcGVyLm1vemlsbGEub3JnL2VuLVVTL2RvY3MvV2ViL0FQSS9GZXRjaF9BUEkpLlxuICpcbiAqIElmIGBmZXRjaGAgaXMgZGVmaW5lZCBvbiB0aGUgZ2xvYmFsIG9iamVjdCAoYHdpbmRvd2AsIGBwcm9jZXNzYCwgZXRjLiksXG4gKiBgdGYudXRpbC5mZXRjaGAgcmV0dXJucyB0aGF0IGZ1bmN0aW9uLlxuICpcbiAqIElmIG5vdCwgYHRmLnV0aWwuZmV0Y2hgIHJldHVybnMgYSBwbGF0Zm9ybS1zcGVjaWZpYyBzb2x1dGlvbi5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmVzb3VyY2UgPSBhd2FpdCB0Zi51dGlsLmZldGNoKCdodHRwczovL3VucGtnLmNvbS9AdGVuc29yZmxvdy90ZmpzJyk7XG4gKiAvLyBoYW5kbGUgcmVzcG9uc2VcbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdVdGlsJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGZldGNoKFxuICAgIHBhdGg6IHN0cmluZywgcmVxdWVzdEluaXRzPzogUmVxdWVzdEluaXQpOiBQcm9taXNlPFJlc3BvbnNlPiB7XG4gIHJldHVybiBlbnYoKS5wbGF0Zm9ybS5mZXRjaChwYXRoLCByZXF1ZXN0SW5pdHMpO1xufVxuXG4vKipcbiAqIEVuY29kZXMgdGhlIHByb3ZpZGVkIHN0cmluZyBpbnRvIGJ5dGVzIHVzaW5nIHRoZSBwcm92aWRlZCBlbmNvZGluZyBzY2hlbWUuXG4gKlxuICogQHBhcmFtIHMgVGhlIHN0cmluZyB0byBlbmNvZGUuXG4gKiBAcGFyYW0gZW5jb2RpbmcgVGhlIGVuY29kaW5nIHNjaGVtZS4gRGVmYXVsdHMgdG8gdXRmLTguXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1V0aWwnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlU3RyaW5nKHM6IHN0cmluZywgZW5jb2RpbmcgPSAndXRmLTgnKTogVWludDhBcnJheSB7XG4gIGVuY29kaW5nID0gZW5jb2RpbmcgfHwgJ3V0Zi04JztcbiAgcmV0dXJuIGVudigpLnBsYXRmb3JtLmVuY29kZShzLCBlbmNvZGluZyk7XG59XG5cbi8qKlxuICogRGVjb2RlcyB0aGUgcHJvdmlkZWQgYnl0ZXMgaW50byBhIHN0cmluZyB1c2luZyB0aGUgcHJvdmlkZWQgZW5jb2Rpbmcgc2NoZW1lLlxuICogQHBhcmFtIGJ5dGVzIFRoZSBieXRlcyB0byBkZWNvZGUuXG4gKlxuICogQHBhcmFtIGVuY29kaW5nIFRoZSBlbmNvZGluZyBzY2hlbWUuIERlZmF1bHRzIHRvIHV0Zi04LlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdVdGlsJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZVN0cmluZyhieXRlczogVWludDhBcnJheSwgZW5jb2RpbmcgPSAndXRmLTgnKTogc3RyaW5nIHtcbiAgZW5jb2RpbmcgPSBlbmNvZGluZyB8fCAndXRmLTgnO1xuICByZXR1cm4gZW52KCkucGxhdGZvcm0uZGVjb2RlKGJ5dGVzLCBlbmNvZGluZyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1R5cGVkQXJyYXkoYToge30pOiBhIGlzIEZsb2F0MzJBcnJheXxJbnQzMkFycmF5fFVpbnQ4QXJyYXl8XG4gICAgVWludDhDbGFtcGVkQXJyYXkge1xuICByZXR1cm4gZW52KCkucGxhdGZvcm0uaXNUeXBlZEFycmF5KGEpO1xufVxuXG4vLyBOT1RFOiBXZSBleHBsaWNpdGx5IHR5cGUgb3V0IHdoYXQgVCBleHRlbmRzIGluc3RlYWQgb2YgYW55IHNvIHRoYXRcbi8vIHV0aWwuZmxhdHRlbiBvbiBhIG5lc3RlZCBhcnJheSBvZiBudW1iZXIgZG9lc24ndCB0cnkgdG8gaW5mZXIgVCBhcyBhXG4vLyBudW1iZXJbXVtdLCBjYXVzaW5nIHVzIHRvIGV4cGxpY2l0bHkgdHlwZSB1dGlsLmZsYXR0ZW48bnVtYmVyPigpLlxuLyoqXG4gKiAgRmxhdHRlbnMgYW4gYXJiaXRyYXJpbHkgbmVzdGVkIGFycmF5LlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gW1sxLCAyXSwgWzMsIDRdLCBbNSwgWzYsIFs3XV1dXTtcbiAqIGNvbnN0IGZsYXQgPSB0Zi51dGlsLmZsYXR0ZW4oYSk7XG4gKiBjb25zb2xlLmxvZyhmbGF0KTtcbiAqIGBgYFxuICpcbiAqICBAcGFyYW0gYXJyIFRoZSBuZXN0ZWQgYXJyYXkgdG8gZmxhdHRlbi5cbiAqICBAcGFyYW0gcmVzdWx0IFRoZSBkZXN0aW5hdGlvbiBhcnJheSB3aGljaCBob2xkcyB0aGUgZWxlbWVudHMuXG4gKiAgQHBhcmFtIHNraXBUeXBlZEFycmF5IElmIHRydWUsIGF2b2lkcyBmbGF0dGVuaW5nIHRoZSB0eXBlZCBhcnJheXMuIERlZmF1bHRzXG4gKiAgICAgIHRvIGZhbHNlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdVdGlsJywgbmFtZXNwYWNlOiAndXRpbCd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvblxuZmxhdHRlbjxUIGV4dGVuZHMgbnVtYmVyfGJvb2xlYW58c3RyaW5nfFByb21pc2U8bnVtYmVyPnxUeXBlZEFycmF5PihcbiAgICBhcnI6IFR8UmVjdXJzaXZlQXJyYXk8VD4sIHJlc3VsdDogVFtdID0gW10sIHNraXBUeXBlZEFycmF5ID0gZmFsc2UpOiBUW10ge1xuICBpZiAocmVzdWx0ID09IG51bGwpIHtcbiAgICByZXN1bHQgPSBbXTtcbiAgfVxuICBpZiAodHlwZW9mIGFyciA9PT0gJ2Jvb2xlYW4nIHx8IHR5cGVvZiBhcnIgPT09ICdudW1iZXInIHx8XG4gICAgdHlwZW9mIGFyciA9PT0gJ3N0cmluZycgfHwgYmFzZS5pc1Byb21pc2UoYXJyKSB8fCBhcnIgPT0gbnVsbCB8fFxuICAgICAgaXNUeXBlZEFycmF5KGFycikgJiYgc2tpcFR5cGVkQXJyYXkpIHtcbiAgICByZXN1bHQucHVzaChhcnIgYXMgVCk7XG4gIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheShhcnIpIHx8IGlzVHlwZWRBcnJheShhcnIpKSB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhcnIubGVuZ3RoOyArK2kpIHtcbiAgICAgIGZsYXR0ZW4oYXJyW2ldLCByZXN1bHQsIHNraXBUeXBlZEFycmF5KTtcbiAgICB9XG4gIH0gZWxzZSB7XG4gICAgbGV0IG1heEluZGV4ID0gLTE7XG4gICAgZm9yIChjb25zdCBrZXkgb2YgT2JqZWN0LmtleXMoYXJyKSkge1xuICAgICAgLy8gMCBvciBwb3NpdGl2ZSBpbnRlZ2VyLlxuICAgICAgaWYgKC9eKFsxLTldK1swLTldKnwwKSQvLnRlc3Qoa2V5KSkge1xuICAgICAgICBtYXhJbmRleCA9IE1hdGgubWF4KG1heEluZGV4LCBOdW1iZXIoa2V5KSk7XG4gICAgICB9XG4gICAgfVxuICAgIGZvciAobGV0IGkgPSAwOyBpIDw9IG1heEluZGV4OyBpKyspIHtcbiAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgICAgIGZsYXR0ZW4oKGFyciBhcyBSZWN1cnNpdmVBcnJheTxUPilbaV0sIHJlc3VsdCwgc2tpcFR5cGVkQXJyYXkpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmVzdWx0O1xufVxuIl19