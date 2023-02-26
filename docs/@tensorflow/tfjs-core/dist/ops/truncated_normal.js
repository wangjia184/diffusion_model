/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { assertNonNegativeIntegerDimensions } from '../util_base';
import { buffer } from './buffer';
import { op } from './operation';
import { MPRandGauss } from './rand_util';
/**
 * Creates a `tf.Tensor` with values sampled from a truncated normal
 * distribution.
 *
 * ```js
 * tf.truncatedNormal([2, 2]).print();
 * ```
 *
 * The generated values follow a normal distribution with specified mean and
 * standard deviation, except that values whose magnitude is more than 2
 * standard deviations from the mean are dropped and re-picked.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output tensor.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function truncatedNormal_(shape, mean = 0, stdDev = 1, dtype, seed) {
    assertNonNegativeIntegerDimensions(shape);
    if (dtype != null && dtype === 'bool') {
        throw new Error(`Unsupported data type $ { dtype }`);
    }
    const randGauss = new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    const res = buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
        res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
}
export const truncatedNormal = /* @__PURE__ */ op({ truncatedNormal_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJ1bmNhdGVkX25vcm1hbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3RydW5jYXRlZF9ub3JtYWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUgsT0FBTyxFQUFDLGtDQUFrQyxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhFLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRXhDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBbUJHO0FBQ0gsU0FBUyxnQkFBZ0IsQ0FDckIsS0FBa0IsRUFBRSxJQUFJLEdBQUcsQ0FBQyxFQUFFLE1BQU0sR0FBRyxDQUFDLEVBQUUsS0FBeUIsRUFDbkUsSUFBYTtJQUNmLGtDQUFrQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzFDLElBQUksS0FBSyxJQUFJLElBQUksSUFBSyxLQUFrQixLQUFLLE1BQU0sRUFBRTtRQUNuRCxNQUFNLElBQUksS0FBSyxDQUFDLG1DQUFtQyxDQUFDLENBQUM7S0FDdEQ7SUFDRCxNQUFNLFNBQVMsR0FDWCxJQUFJLFdBQVcsQ0FBQyxJQUFJLEVBQUUsTUFBTSxFQUFFLEtBQUssRUFBRSxJQUFJLENBQUMsZUFBZSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDakMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQzFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsU0FBUyxDQUFDLFNBQVMsRUFBRSxDQUFDO0tBQ3ZDO0lBQ0QsT0FBTyxHQUFHLENBQUMsUUFBUSxFQUFFLENBQUM7QUFDeEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGVBQWUsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsZ0JBQWdCLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7RGF0YVR5cGUsIFJhbmssIFNoYXBlTWFwfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydE5vbk5lZ2F0aXZlSW50ZWdlckRpbWVuc2lvbnN9IGZyb20gJy4uL3V0aWxfYmFzZSc7XG5cbmltcG9ydCB7YnVmZmVyfSBmcm9tICcuL2J1ZmZlcic7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge01QUmFuZEdhdXNzfSBmcm9tICcuL3JhbmRfdXRpbCc7XG5cbi8qKlxuICogQ3JlYXRlcyBhIGB0Zi5UZW5zb3JgIHdpdGggdmFsdWVzIHNhbXBsZWQgZnJvbSBhIHRydW5jYXRlZCBub3JtYWxcbiAqIGRpc3RyaWJ1dGlvbi5cbiAqXG4gKiBgYGBqc1xuICogdGYudHJ1bmNhdGVkTm9ybWFsKFsyLCAyXSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIFRoZSBnZW5lcmF0ZWQgdmFsdWVzIGZvbGxvdyBhIG5vcm1hbCBkaXN0cmlidXRpb24gd2l0aCBzcGVjaWZpZWQgbWVhbiBhbmRcbiAqIHN0YW5kYXJkIGRldmlhdGlvbiwgZXhjZXB0IHRoYXQgdmFsdWVzIHdob3NlIG1hZ25pdHVkZSBpcyBtb3JlIHRoYW4gMlxuICogc3RhbmRhcmQgZGV2aWF0aW9ucyBmcm9tIHRoZSBtZWFuIGFyZSBkcm9wcGVkIGFuZCByZS1waWNrZWQuXG4gKlxuICogQHBhcmFtIHNoYXBlIEFuIGFycmF5IG9mIGludGVnZXJzIGRlZmluaW5nIHRoZSBvdXRwdXQgdGVuc29yIHNoYXBlLlxuICogQHBhcmFtIG1lYW4gVGhlIG1lYW4gb2YgdGhlIG5vcm1hbCBkaXN0cmlidXRpb24uXG4gKiBAcGFyYW0gc3RkRGV2IFRoZSBzdGFuZGFyZCBkZXZpYXRpb24gb2YgdGhlIG5vcm1hbCBkaXN0cmlidXRpb24uXG4gKiBAcGFyYW0gZHR5cGUgVGhlIGRhdGEgdHlwZSBvZiB0aGUgb3V0cHV0IHRlbnNvci5cbiAqIEBwYXJhbSBzZWVkIFRoZSBzZWVkIGZvciB0aGUgcmFuZG9tIG51bWJlciBnZW5lcmF0b3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICovXG5mdW5jdGlvbiB0cnVuY2F0ZWROb3JtYWxfPFIgZXh0ZW5kcyBSYW5rPihcbiAgICBzaGFwZTogU2hhcGVNYXBbUl0sIG1lYW4gPSAwLCBzdGREZXYgPSAxLCBkdHlwZT86ICdmbG9hdDMyJ3wnaW50MzInLFxuICAgIHNlZWQ/OiBudW1iZXIpOiBUZW5zb3I8Uj4ge1xuICBhc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zKHNoYXBlKTtcbiAgaWYgKGR0eXBlICE9IG51bGwgJiYgKGR0eXBlIGFzIERhdGFUeXBlKSA9PT0gJ2Jvb2wnKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCBkYXRhIHR5cGUgJCB7IGR0eXBlIH1gKTtcbiAgfVxuICBjb25zdCByYW5kR2F1c3MgPVxuICAgICAgbmV3IE1QUmFuZEdhdXNzKG1lYW4sIHN0ZERldiwgZHR5cGUsIHRydWUgLyogdHJ1bmNhdGVkICovLCBzZWVkKTtcbiAgY29uc3QgcmVzID0gYnVmZmVyKHNoYXBlLCBkdHlwZSk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcmVzLnZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgIHJlcy52YWx1ZXNbaV0gPSByYW5kR2F1c3MubmV4dFZhbHVlKCk7XG4gIH1cbiAgcmV0dXJuIHJlcy50b1RlbnNvcigpO1xufVxuXG5leHBvcnQgY29uc3QgdHJ1bmNhdGVkTm9ybWFsID0gLyogQF9fUFVSRV9fICovIG9wKHt0cnVuY2F0ZWROb3JtYWxffSk7XG4iXX0=