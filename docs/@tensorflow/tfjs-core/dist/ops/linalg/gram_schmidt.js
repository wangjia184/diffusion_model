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
import { assert } from '../../util';
import { div } from '../div';
import { mul } from '../mul';
import { norm } from '../norm';
import { op } from '../operation';
import { split } from '../split';
import { squeeze } from '../squeeze';
import { stack } from '../stack';
import { sub } from '../sub';
import { sum } from '../sum';
/**
 * Gram-Schmidt orthogonalization.
 *
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * let y = tf.linalg.gramSchmidt(x);
 * y.print();
 * console.log('Orthogonalized:');
 * y.dot(y.transpose()).print();  // should be nearly the identity matrix.
 * console.log('First row direction maintained:');
 * const data = await y.array();
 * console.log(data[0][1] / data[0][0]);  // should be nearly 2.
 * ```
 *
 * @param xs The vectors to be orthogonalized, in one of the two following
 *   formats:
 *   - An Array of `tf.Tensor1D`.
 *   - A `tf.Tensor2D`, i.e., a matrix, in which case the vectors are the rows
 *     of `xs`.
 *   In each case, all the vectors must have the same length and the length
 *   must be greater than or equal to the number of vectors.
 * @returns The orthogonalized and normalized vectors or matrix.
 *   Orthogonalization means that the vectors or the rows of the matrix
 *   are orthogonal (zero inner products). Normalization means that each
 *   vector or each row of the matrix has an L2 norm that equals `1`.
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function gramSchmidt_(xs) {
    let inputIsTensor2D;
    if (Array.isArray(xs)) {
        inputIsTensor2D = false;
        assert(xs != null && xs.length > 0, () => 'Gram-Schmidt process: input must not be null, undefined, or ' +
            'empty');
        const dim = xs[0].shape[0];
        for (let i = 1; i < xs.length; ++i) {
            assert(xs[i].shape[0] === dim, () => 'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
                `(${xs[i].shape[0]} vs. ${dim})`);
        }
    }
    else {
        inputIsTensor2D = true;
        xs = split(xs, xs.shape[0], 0).map(x => squeeze(x, [0]));
    }
    assert(xs.length <= xs[0].shape[0], () => `Gram-Schmidt: Number of vectors (${xs.length}) exceeds ` +
        `number of dimensions (${xs[0].shape[0]}).`);
    const ys = [];
    const xs1d = xs;
    for (let i = 0; i < xs.length; ++i) {
        ys.push(ENGINE.tidy(() => {
            let x = xs1d[i];
            if (i > 0) {
                for (let j = 0; j < i; ++j) {
                    const proj = mul(sum(mul(ys[j], x)), ys[j]);
                    x = sub(x, proj);
                }
            }
            return div(x, norm(x, 'euclidean'));
        }));
    }
    if (inputIsTensor2D) {
        return stack(ys, 0);
    }
    else {
        return ys;
    }
}
export const gramSchmidt = /* @__PURE__ */ op({ gramSchmidt_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3JhbV9zY2htaWR0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvbGluYWxnL2dyYW1fc2NobWlkdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRXBDLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFFbEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDN0IsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFM0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTJCRztBQUNILFNBQVMsWUFBWSxDQUFDLEVBQXVCO0lBQzNDLElBQUksZUFBd0IsQ0FBQztJQUM3QixJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLEVBQUU7UUFDckIsZUFBZSxHQUFHLEtBQUssQ0FBQztRQUN4QixNQUFNLENBQ0YsRUFBRSxJQUFJLElBQUksSUFBSSxFQUFFLENBQUMsTUFBTSxHQUFHLENBQUMsRUFDM0IsR0FBRyxFQUFFLENBQUMsOERBQThEO1lBQ2hFLE9BQU8sQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0IsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDbEMsTUFBTSxDQUNGLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssR0FBRyxFQUN0QixHQUFHLEVBQUUsQ0FDRCwrREFBK0Q7Z0JBQy9ELElBQUssRUFBaUIsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLFFBQVEsR0FBRyxHQUFHLENBQUMsQ0FBQztTQUMzRDtLQUNGO1NBQU07UUFDTCxlQUFlLEdBQUcsSUFBSSxDQUFDO1FBQ3ZCLEVBQUUsR0FBRyxLQUFLLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUMxRDtJQUVELE1BQU0sQ0FDRixFQUFFLENBQUMsTUFBTSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzNCLEdBQUcsRUFBRSxDQUFDLG9DQUNLLEVBQWlCLENBQUMsTUFBTSxZQUFZO1FBQzNDLHlCQUEwQixFQUFpQixDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7SUFFckUsTUFBTSxFQUFFLEdBQWUsRUFBRSxDQUFDO0lBQzFCLE1BQU0sSUFBSSxHQUFHLEVBQUUsQ0FBQztJQUNoQixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtRQUNsQyxFQUFFLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ3ZCLElBQUksQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNoQixJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUU7Z0JBQ1QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRTtvQkFDMUIsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQzVDLENBQUMsR0FBRyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO2lCQUNsQjthQUNGO1lBQ0QsT0FBTyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQztRQUN0QyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ0w7SUFFRCxJQUFJLGVBQWUsRUFBRTtRQUNuQixPQUFPLEtBQUssQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFhLENBQUM7S0FDakM7U0FBTTtRQUNMLE9BQU8sRUFBRSxDQUFDO0tBQ1g7QUFDSCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sV0FBVyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxZQUFZLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vLi4vZW5naW5lJztcbmltcG9ydCB7VGVuc29yMUQsIFRlbnNvcjJEfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHthc3NlcnR9IGZyb20gJy4uLy4uL3V0aWwnO1xuXG5pbXBvcnQge2Rpdn0gZnJvbSAnLi4vZGl2JztcbmltcG9ydCB7bXVsfSBmcm9tICcuLi9tdWwnO1xuaW1wb3J0IHtub3JtfSBmcm9tICcuLi9ub3JtJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3NwbGl0fSBmcm9tICcuLi9zcGxpdCc7XG5pbXBvcnQge3NxdWVlemV9IGZyb20gJy4uL3NxdWVlemUnO1xuaW1wb3J0IHtzdGFja30gZnJvbSAnLi4vc3RhY2snO1xuaW1wb3J0IHtzdWJ9IGZyb20gJy4uL3N1Yic7XG5pbXBvcnQge3N1bX0gZnJvbSAnLi4vc3VtJztcblxuLyoqXG4gKiBHcmFtLVNjaG1pZHQgb3J0aG9nb25hbGl6YXRpb24uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbWzEsIDJdLCBbMywgNF1dKTtcbiAqIGxldCB5ID0gdGYubGluYWxnLmdyYW1TY2htaWR0KHgpO1xuICogeS5wcmludCgpO1xuICogY29uc29sZS5sb2coJ09ydGhvZ29uYWxpemVkOicpO1xuICogeS5kb3QoeS50cmFuc3Bvc2UoKSkucHJpbnQoKTsgIC8vIHNob3VsZCBiZSBuZWFybHkgdGhlIGlkZW50aXR5IG1hdHJpeC5cbiAqIGNvbnNvbGUubG9nKCdGaXJzdCByb3cgZGlyZWN0aW9uIG1haW50YWluZWQ6Jyk7XG4gKiBjb25zdCBkYXRhID0gYXdhaXQgeS5hcnJheSgpO1xuICogY29uc29sZS5sb2coZGF0YVswXVsxXSAvIGRhdGFbMF1bMF0pOyAgLy8gc2hvdWxkIGJlIG5lYXJseSAyLlxuICogYGBgXG4gKlxuICogQHBhcmFtIHhzIFRoZSB2ZWN0b3JzIHRvIGJlIG9ydGhvZ29uYWxpemVkLCBpbiBvbmUgb2YgdGhlIHR3byBmb2xsb3dpbmdcbiAqICAgZm9ybWF0czpcbiAqICAgLSBBbiBBcnJheSBvZiBgdGYuVGVuc29yMURgLlxuICogICAtIEEgYHRmLlRlbnNvcjJEYCwgaS5lLiwgYSBtYXRyaXgsIGluIHdoaWNoIGNhc2UgdGhlIHZlY3RvcnMgYXJlIHRoZSByb3dzXG4gKiAgICAgb2YgYHhzYC5cbiAqICAgSW4gZWFjaCBjYXNlLCBhbGwgdGhlIHZlY3RvcnMgbXVzdCBoYXZlIHRoZSBzYW1lIGxlbmd0aCBhbmQgdGhlIGxlbmd0aFxuICogICBtdXN0IGJlIGdyZWF0ZXIgdGhhbiBvciBlcXVhbCB0byB0aGUgbnVtYmVyIG9mIHZlY3RvcnMuXG4gKiBAcmV0dXJucyBUaGUgb3J0aG9nb25hbGl6ZWQgYW5kIG5vcm1hbGl6ZWQgdmVjdG9ycyBvciBtYXRyaXguXG4gKiAgIE9ydGhvZ29uYWxpemF0aW9uIG1lYW5zIHRoYXQgdGhlIHZlY3RvcnMgb3IgdGhlIHJvd3Mgb2YgdGhlIG1hdHJpeFxuICogICBhcmUgb3J0aG9nb25hbCAoemVybyBpbm5lciBwcm9kdWN0cykuIE5vcm1hbGl6YXRpb24gbWVhbnMgdGhhdCBlYWNoXG4gKiAgIHZlY3RvciBvciBlYWNoIHJvdyBvZiB0aGUgbWF0cml4IGhhcyBhbiBMMiBub3JtIHRoYXQgZXF1YWxzIGAxYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOidPcGVyYXRpb25zJywgc3ViaGVhZGluZzonTGluZWFyIEFsZ2VicmEnLCBuYW1lc3BhY2U6J2xpbmFsZyd9XG4gKi9cbmZ1bmN0aW9uIGdyYW1TY2htaWR0Xyh4czogVGVuc29yMURbXXxUZW5zb3IyRCk6IFRlbnNvcjFEW118VGVuc29yMkQge1xuICBsZXQgaW5wdXRJc1RlbnNvcjJEOiBib29sZWFuO1xuICBpZiAoQXJyYXkuaXNBcnJheSh4cykpIHtcbiAgICBpbnB1dElzVGVuc29yMkQgPSBmYWxzZTtcbiAgICBhc3NlcnQoXG4gICAgICAgIHhzICE9IG51bGwgJiYgeHMubGVuZ3RoID4gMCxcbiAgICAgICAgKCkgPT4gJ0dyYW0tU2NobWlkdCBwcm9jZXNzOiBpbnB1dCBtdXN0IG5vdCBiZSBudWxsLCB1bmRlZmluZWQsIG9yICcgK1xuICAgICAgICAgICAgJ2VtcHR5Jyk7XG4gICAgY29uc3QgZGltID0geHNbMF0uc2hhcGVbMF07XG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB4cy5sZW5ndGg7ICsraSkge1xuICAgICAgYXNzZXJ0KFxuICAgICAgICAgIHhzW2ldLnNoYXBlWzBdID09PSBkaW0sXG4gICAgICAgICAgKCkgPT5cbiAgICAgICAgICAgICAgJ0dyYW0tU2NobWlkdDogTm9uLXVuaXF1ZSBsZW5ndGhzIGZvdW5kIGluIHRoZSBpbnB1dCB2ZWN0b3JzOiAnICtcbiAgICAgICAgICAgICAgYCgkeyh4cyBhcyBUZW5zb3IxRFtdKVtpXS5zaGFwZVswXX0gdnMuICR7ZGltfSlgKTtcbiAgICB9XG4gIH0gZWxzZSB7XG4gICAgaW5wdXRJc1RlbnNvcjJEID0gdHJ1ZTtcbiAgICB4cyA9IHNwbGl0KHhzLCB4cy5zaGFwZVswXSwgMCkubWFwKHggPT4gc3F1ZWV6ZSh4LCBbMF0pKTtcbiAgfVxuXG4gIGFzc2VydChcbiAgICAgIHhzLmxlbmd0aCA8PSB4c1swXS5zaGFwZVswXSxcbiAgICAgICgpID0+IGBHcmFtLVNjaG1pZHQ6IE51bWJlciBvZiB2ZWN0b3JzICgke1xuICAgICAgICAgICAgICAgICh4cyBhcyBUZW5zb3IxRFtdKS5sZW5ndGh9KSBleGNlZWRzIGAgK1xuICAgICAgICAgIGBudW1iZXIgb2YgZGltZW5zaW9ucyAoJHsoeHMgYXMgVGVuc29yMURbXSlbMF0uc2hhcGVbMF19KS5gKTtcblxuICBjb25zdCB5czogVGVuc29yMURbXSA9IFtdO1xuICBjb25zdCB4czFkID0geHM7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgeHMubGVuZ3RoOyArK2kpIHtcbiAgICB5cy5wdXNoKEVOR0lORS50aWR5KCgpID0+IHtcbiAgICAgIGxldCB4ID0geHMxZFtpXTtcbiAgICAgIGlmIChpID4gMCkge1xuICAgICAgICBmb3IgKGxldCBqID0gMDsgaiA8IGk7ICsraikge1xuICAgICAgICAgIGNvbnN0IHByb2ogPSBtdWwoc3VtKG11bCh5c1tqXSwgeCkpLCB5c1tqXSk7XG4gICAgICAgICAgeCA9IHN1Yih4LCBwcm9qKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgcmV0dXJuIGRpdih4LCBub3JtKHgsICdldWNsaWRlYW4nKSk7XG4gICAgfSkpO1xuICB9XG5cbiAgaWYgKGlucHV0SXNUZW5zb3IyRCkge1xuICAgIHJldHVybiBzdGFjayh5cywgMCkgYXMgVGVuc29yMkQ7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIHlzO1xuICB9XG59XG5cbmV4cG9ydCBjb25zdCBncmFtU2NobWlkdCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7Z3JhbVNjaG1pZHRffSk7XG4iXX0=