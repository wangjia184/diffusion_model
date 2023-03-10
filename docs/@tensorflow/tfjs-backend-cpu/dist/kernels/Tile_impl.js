/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { buffer } from '@tensorflow/tfjs-core';
/**
 * An implementation of the tile kernel shared between webgl and cpu for string
 * tensors only.
 */
export function tileImpl(xBuf, reps) {
    const newShape = new Array(xBuf.rank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = xBuf.shape[i] * reps[i];
    }
    const result = buffer(newShape, xBuf.dtype);
    for (let i = 0; i < result.values.length; ++i) {
        const newLoc = result.indexToLoc(i);
        const originalLoc = new Array(xBuf.rank);
        for (let j = 0; j < originalLoc.length; j++) {
            originalLoc[j] = newLoc[j] % xBuf.shape[j];
        }
        const originalIndex = xBuf.locToIndex(originalLoc);
        result.values[i] = xBuf.values[originalIndex];
    }
    return result;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiVGlsZV9pbXBsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9UaWxlX2ltcGwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBK0IsTUFBTSx1QkFBdUIsQ0FBQztBQUUzRTs7O0dBR0c7QUFFSCxNQUFNLFVBQVUsUUFBUSxDQUNwQixJQUErQixFQUMvQixJQUFjO0lBQ2hCLE1BQU0sUUFBUSxHQUFhLElBQUksS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUNoRCxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsUUFBUSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUN4QyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDdkM7SUFDRCxNQUFNLE1BQU0sR0FBRyxNQUFNLENBQUMsUUFBUSxFQUFFLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUM1QyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDN0MsTUFBTSxNQUFNLEdBQUcsTUFBTSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVwQyxNQUFNLFdBQVcsR0FBYSxJQUFJLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDbkQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDM0MsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQzVDO1FBRUQsTUFBTSxhQUFhLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUVuRCxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsYUFBYSxDQUFDLENBQUM7S0FDL0M7SUFDRCxPQUFPLE1BQW1DLENBQUM7QUFDN0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtidWZmZXIsIERhdGFUeXBlLCBSYW5rLCBUZW5zb3JCdWZmZXJ9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbi8qKlxuICogQW4gaW1wbGVtZW50YXRpb24gb2YgdGhlIHRpbGUga2VybmVsIHNoYXJlZCBiZXR3ZWVuIHdlYmdsIGFuZCBjcHUgZm9yIHN0cmluZ1xuICogdGVuc29ycyBvbmx5LlxuICovXG5cbmV4cG9ydCBmdW5jdGlvbiB0aWxlSW1wbDxSIGV4dGVuZHMgUmFuaz4oXG4gICAgeEJ1ZjogVGVuc29yQnVmZmVyPFIsIERhdGFUeXBlPixcbiAgICByZXBzOiBudW1iZXJbXSk6IFRlbnNvckJ1ZmZlcjxSLCBEYXRhVHlwZT4ge1xuICBjb25zdCBuZXdTaGFwZTogbnVtYmVyW10gPSBuZXcgQXJyYXkoeEJ1Zi5yYW5rKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZXdTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgIG5ld1NoYXBlW2ldID0geEJ1Zi5zaGFwZVtpXSAqIHJlcHNbaV07XG4gIH1cbiAgY29uc3QgcmVzdWx0ID0gYnVmZmVyKG5ld1NoYXBlLCB4QnVmLmR0eXBlKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXN1bHQudmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgbmV3TG9jID0gcmVzdWx0LmluZGV4VG9Mb2MoaSk7XG5cbiAgICBjb25zdCBvcmlnaW5hbExvYzogbnVtYmVyW10gPSBuZXcgQXJyYXkoeEJ1Zi5yYW5rKTtcbiAgICBmb3IgKGxldCBqID0gMDsgaiA8IG9yaWdpbmFsTG9jLmxlbmd0aDsgaisrKSB7XG4gICAgICBvcmlnaW5hbExvY1tqXSA9IG5ld0xvY1tqXSAlIHhCdWYuc2hhcGVbal07XG4gICAgfVxuXG4gICAgY29uc3Qgb3JpZ2luYWxJbmRleCA9IHhCdWYubG9jVG9JbmRleChvcmlnaW5hbExvYyk7XG5cbiAgICByZXN1bHQudmFsdWVzW2ldID0geEJ1Zi52YWx1ZXNbb3JpZ2luYWxJbmRleF07XG4gIH1cbiAgcmV0dXJuIHJlc3VsdCBhcyBUZW5zb3JCdWZmZXI8UiwgRGF0YVR5cGU+O1xufVxuIl19