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
import { ENGINE } from '../../engine';
import { IFFT } from '../../kernel_names';
import { assert } from '../../util';
import { op } from '../operation';
/**
 * Inverse fast Fourier transform.
 *
 * Computes the inverse 1-dimensional discrete Fourier transform over the
 * inner-most dimension of input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([1, 2, 3]);
 * const x = tf.complex(real, imag);
 *
 * x.ifft().print();  // tf.spectral.ifft(x).print();
 * ```
 * @param input The complex input to compute an ifft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function ifft_(input) {
    assert(input.dtype === 'complex64', () => `The dtype for tf.spectral.ifft() must be complex64 ` +
        `but got ${input.dtype}.`);
    const inputs = { input };
    return ENGINE.runKernel(IFFT, inputs);
}
export const ifft = /* @__PURE__ */ op({ ifft_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaWZmdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NwZWN0cmFsL2lmZnQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNwQyxPQUFPLEVBQUMsSUFBSSxFQUFhLE1BQU0sb0JBQW9CLENBQUM7QUFHcEQsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNsQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhDOzs7Ozs7Ozs7Ozs7Ozs7O0dBZ0JHO0FBQ0gsU0FBUyxLQUFLLENBQUMsS0FBYTtJQUMxQixNQUFNLENBQ0YsS0FBSyxDQUFDLEtBQUssS0FBSyxXQUFXLEVBQzNCLEdBQUcsRUFBRSxDQUFDLHFEQUFxRDtRQUN2RCxXQUFXLEtBQUssQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDO0lBRW5DLE1BQU0sTUFBTSxHQUFlLEVBQUMsS0FBSyxFQUFDLENBQUM7SUFFbkMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLElBQUksRUFBRSxNQUFtQyxDQUFDLENBQUM7QUFDckUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsS0FBSyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uLy4uL2VuZ2luZSc7XG5pbXBvcnQge0lGRlQsIElGRlRJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2Fzc2VydH0gZnJvbSAnLi4vLi4vdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEludmVyc2UgZmFzdCBGb3VyaWVyIHRyYW5zZm9ybS5cbiAqXG4gKiBDb21wdXRlcyB0aGUgaW52ZXJzZSAxLWRpbWVuc2lvbmFsIGRpc2NyZXRlIEZvdXJpZXIgdHJhbnNmb3JtIG92ZXIgdGhlXG4gKiBpbm5lci1tb3N0IGRpbWVuc2lvbiBvZiBpbnB1dC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmVhbCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKiBjb25zdCBpbWFnID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAqIGNvbnN0IHggPSB0Zi5jb21wbGV4KHJlYWwsIGltYWcpO1xuICpcbiAqIHguaWZmdCgpLnByaW50KCk7ICAvLyB0Zi5zcGVjdHJhbC5pZmZ0KHgpLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSBpbnB1dCBUaGUgY29tcGxleCBpbnB1dCB0byBjb21wdXRlIGFuIGlmZnQgb3Zlci5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdTcGVjdHJhbCcsIG5hbWVzcGFjZTogJ3NwZWN0cmFsJ31cbiAqL1xuZnVuY3Rpb24gaWZmdF8oaW5wdXQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIGFzc2VydChcbiAgICAgIGlucHV0LmR0eXBlID09PSAnY29tcGxleDY0JyxcbiAgICAgICgpID0+IGBUaGUgZHR5cGUgZm9yIHRmLnNwZWN0cmFsLmlmZnQoKSBtdXN0IGJlIGNvbXBsZXg2NCBgICtcbiAgICAgICAgICBgYnV0IGdvdCAke2lucHV0LmR0eXBlfS5gKTtcblxuICBjb25zdCBpbnB1dHM6IElGRlRJbnB1dHMgPSB7aW5wdXR9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKElGRlQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGlmZnQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2lmZnRffSk7XG4iXX0=