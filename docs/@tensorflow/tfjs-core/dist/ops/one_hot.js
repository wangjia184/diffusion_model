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
import { ENGINE } from '../engine';
import { OneHot } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Creates a one-hot `tf.Tensor`. The locations represented by `indices` take
 * value `onValue` (defaults to 1), while all other locations take value
 * `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank
 * `R+1` with the last axis of size `depth`.
 * `indices` used to encode prediction class must start from 0. For example,
 *  if you have 3 classes of data, class 1 should be encoded as 0, class 2
 *  should be 1, and class 3 should be 2.
 *
 * ```js
 * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
 * ```
 *
 * @param indices `tf.Tensor` of indices with dtype `int32`. Indices must
 * start from 0.
 * @param depth The depth of the one hot dimension.
 * @param onValue A number used to fill in the output when the index matches
 * the location.
 * @param offValue A number used to fill in the output when the index does
 *     not match the location.
 * @param dtype The dtype of the output tensor, default to 'int32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function oneHot_(indices, depth, onValue = 1, offValue = 0, dtype = 'int32') {
    if (depth < 2) {
        throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
    }
    const $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
    const inputs = { indices: $indices };
    const attrs = { dtype, depth, onValue, offValue };
    return ENGINE.runKernel(OneHot, inputs, attrs);
}
export const oneHot = /* @__PURE__ */ op({ oneHot_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib25lX2hvdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL29uZV9ob3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsTUFBTSxFQUE0QixNQUFNLGlCQUFpQixDQUFDO0FBSWxFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXVCRztBQUNILFNBQVMsT0FBTyxDQUNaLE9BQTBCLEVBQUUsS0FBYSxFQUFFLE9BQU8sR0FBRyxDQUFDLEVBQUUsUUFBUSxHQUFHLENBQUMsRUFDcEUsUUFBa0IsT0FBTztJQUMzQixJQUFJLEtBQUssR0FBRyxDQUFDLEVBQUU7UUFDYixNQUFNLElBQUksS0FBSyxDQUFDLGlEQUFpRCxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzNFO0lBQ0QsTUFBTSxRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRXhFLE1BQU0sTUFBTSxHQUFpQixFQUFDLE9BQU8sRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUNqRCxNQUFNLEtBQUssR0FBZ0IsRUFBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUU3RCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLE1BQU0sRUFBRSxNQUFtQyxFQUMzQyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsT0FBTyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge09uZUhvdCwgT25lSG90QXR0cnMsIE9uZUhvdElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge0RhdGFUeXBlLCBUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDcmVhdGVzIGEgb25lLWhvdCBgdGYuVGVuc29yYC4gVGhlIGxvY2F0aW9ucyByZXByZXNlbnRlZCBieSBgaW5kaWNlc2AgdGFrZVxuICogdmFsdWUgYG9uVmFsdWVgIChkZWZhdWx0cyB0byAxKSwgd2hpbGUgYWxsIG90aGVyIGxvY2F0aW9ucyB0YWtlIHZhbHVlXG4gKiBgb2ZmVmFsdWVgIChkZWZhdWx0cyB0byAwKS4gSWYgYGluZGljZXNgIGlzIHJhbmsgYFJgLCB0aGUgb3V0cHV0IGhhcyByYW5rXG4gKiBgUisxYCB3aXRoIHRoZSBsYXN0IGF4aXMgb2Ygc2l6ZSBgZGVwdGhgLlxuICogYGluZGljZXNgIHVzZWQgdG8gZW5jb2RlIHByZWRpY3Rpb24gY2xhc3MgbXVzdCBzdGFydCBmcm9tIDAuIEZvciBleGFtcGxlLFxuICogIGlmIHlvdSBoYXZlIDMgY2xhc3NlcyBvZiBkYXRhLCBjbGFzcyAxIHNob3VsZCBiZSBlbmNvZGVkIGFzIDAsIGNsYXNzIDJcbiAqICBzaG91bGQgYmUgMSwgYW5kIGNsYXNzIDMgc2hvdWxkIGJlIDIuXG4gKlxuICogYGBganNcbiAqIHRmLm9uZUhvdCh0Zi50ZW5zb3IxZChbMCwgMV0sICdpbnQzMicpLCAzKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIGluZGljZXMgYHRmLlRlbnNvcmAgb2YgaW5kaWNlcyB3aXRoIGR0eXBlIGBpbnQzMmAuIEluZGljZXMgbXVzdFxuICogc3RhcnQgZnJvbSAwLlxuICogQHBhcmFtIGRlcHRoIFRoZSBkZXB0aCBvZiB0aGUgb25lIGhvdCBkaW1lbnNpb24uXG4gKiBAcGFyYW0gb25WYWx1ZSBBIG51bWJlciB1c2VkIHRvIGZpbGwgaW4gdGhlIG91dHB1dCB3aGVuIHRoZSBpbmRleCBtYXRjaGVzXG4gKiB0aGUgbG9jYXRpb24uXG4gKiBAcGFyYW0gb2ZmVmFsdWUgQSBudW1iZXIgdXNlZCB0byBmaWxsIGluIHRoZSBvdXRwdXQgd2hlbiB0aGUgaW5kZXggZG9lc1xuICogICAgIG5vdCBtYXRjaCB0aGUgbG9jYXRpb24uXG4gKiBAcGFyYW0gZHR5cGUgVGhlIGR0eXBlIG9mIHRoZSBvdXRwdXQgdGVuc29yLCBkZWZhdWx0IHRvICdpbnQzMicuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICovXG5mdW5jdGlvbiBvbmVIb3RfKFxuICAgIGluZGljZXM6IFRlbnNvcnxUZW5zb3JMaWtlLCBkZXB0aDogbnVtYmVyLCBvblZhbHVlID0gMSwgb2ZmVmFsdWUgPSAwLFxuICAgIGR0eXBlOiBEYXRhVHlwZSA9ICdpbnQzMicpOiBUZW5zb3Ige1xuICBpZiAoZGVwdGggPCAyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBFcnJvciBpbiBvbmVIb3Q6IGRlcHRoIG11c3QgYmUgPj0yLCBidXQgaXQgaXMgJHtkZXB0aH1gKTtcbiAgfVxuICBjb25zdCAkaW5kaWNlcyA9IGNvbnZlcnRUb1RlbnNvcihpbmRpY2VzLCAnaW5kaWNlcycsICdvbmVIb3QnLCAnaW50MzInKTtcblxuICBjb25zdCBpbnB1dHM6IE9uZUhvdElucHV0cyA9IHtpbmRpY2VzOiAkaW5kaWNlc307XG4gIGNvbnN0IGF0dHJzOiBPbmVIb3RBdHRycyA9IHtkdHlwZSwgZGVwdGgsIG9uVmFsdWUsIG9mZlZhbHVlfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIE9uZUhvdCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBvbmVIb3QgPSAvKiBAX19QVVJFX18gKi8gb3Aoe29uZUhvdF99KTtcbiJdfQ==