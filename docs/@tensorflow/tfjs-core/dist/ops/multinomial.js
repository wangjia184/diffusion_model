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
import { Multinomial } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Creates a `tf.Tensor` with values drawn from a multinomial distribution.
 *
 * ```js
 * const probs = tf.tensor([.75, .25]);
 * tf.multinomial(probs, 3).print();
 * ```
 *
 * @param logits 1D array with unnormalized log-probabilities, or
 *     2D array of shape `[batchSize, numOutcomes]`. See the `normalized`
 *     parameter.
 * @param numSamples Number of samples to draw for each row slice.
 * @param seed The seed number.
 * @param normalized Whether the provided `logits` are normalized true
 *     probabilities (sum to 1). Defaults to false.
 * @return 1D array of shape `[numSamples]`, or 2D array of shape
 *     `[batchSize, numSamples]`, depending on the rank of the input.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function multinomial_(logits, numSamples, seed, normalized = false) {
    const $logits = convertToTensor(logits, 'logits', 'multinomial');
    const numOutcomes = $logits.size;
    const origRank = $logits.rank;
    if (numOutcomes < 2) {
        throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ` +
            `${numOutcomes}.`);
    }
    if (origRank > 2) {
        throw new Error(`Rank of probabilities must be 1 or 2, but is ${origRank}`);
    }
    // TODO(lina128): Investigate correct seed behavior. The code seems not allow
    // setting see to 0.
    seed = seed || Math.random();
    // The kernel only accepts (and returns) rank 2 tensors.
    const logits2D = origRank === 1 ? reshape($logits, [1, -1]) : $logits;
    const inputs = { logits: logits2D };
    const attrs = { numSamples, seed, normalized };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(Multinomial, inputs, attrs);
    // tslint:disable-next-line:no-unnecessary-type-assertion
    return origRank === 1 ? reshape(res, [res.size]) : res;
}
export const multinomial = /* @__PURE__ */ op({ multinomial_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibXVsdGlub21pYWwuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9tdWx0aW5vbWlhbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxXQUFXLEVBQXNDLE1BQU0saUJBQWlCLENBQUM7QUFJakYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBR25ELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVsQzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW1CRztBQUNILFNBQVMsWUFBWSxDQUNqQixNQUFvQyxFQUFFLFVBQWtCLEVBQUUsSUFBYSxFQUN2RSxVQUFVLEdBQUcsS0FBSztJQUNwQixNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxhQUFhLENBQUMsQ0FBQztJQUNqRSxNQUFNLFdBQVcsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDO0lBQ2pDLE1BQU0sUUFBUSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUM7SUFDOUIsSUFBSSxXQUFXLEdBQUcsQ0FBQyxFQUFFO1FBQ25CLE1BQU0sSUFBSSxLQUFLLENBQ1gsOERBQThEO1lBQzlELEdBQUcsV0FBVyxHQUFHLENBQUMsQ0FBQztLQUN4QjtJQUNELElBQUksUUFBUSxHQUFHLENBQUMsRUFBRTtRQUNoQixNQUFNLElBQUksS0FBSyxDQUFDLGdEQUFnRCxRQUFRLEVBQUUsQ0FBQyxDQUFDO0tBQzdFO0lBQ0QsNkVBQTZFO0lBQzdFLG9CQUFvQjtJQUNwQixJQUFJLEdBQUcsSUFBSSxJQUFJLElBQUksQ0FBQyxNQUFNLEVBQUUsQ0FBQztJQUU3Qix3REFBd0Q7SUFDeEQsTUFBTSxRQUFRLEdBQ1YsUUFBUSxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQW1CLENBQUM7SUFFckUsTUFBTSxNQUFNLEdBQXNCLEVBQUMsTUFBTSxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQ3JELE1BQU0sS0FBSyxHQUFxQixFQUFDLFVBQVUsRUFBRSxJQUFJLEVBQUUsVUFBVSxFQUFDLENBQUM7SUFFL0QsMERBQTBEO0lBQzFELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1osV0FBVyxFQUFFLE1BQW1DLEVBQ2hELEtBQWdDLENBQWEsQ0FBQztJQUU5RCx5REFBeUQ7SUFDekQsT0FBTyxRQUFRLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFhLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQztBQUNyRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sV0FBVyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxZQUFZLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7TXVsdGlub21pYWwsIE11bHRpbm9taWFsQXR0cnMsIE11bHRpbm9taWFsSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjFELCBUZW5zb3IyRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5cbi8qKlxuICogQ3JlYXRlcyBhIGB0Zi5UZW5zb3JgIHdpdGggdmFsdWVzIGRyYXduIGZyb20gYSBtdWx0aW5vbWlhbCBkaXN0cmlidXRpb24uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHByb2JzID0gdGYudGVuc29yKFsuNzUsIC4yNV0pO1xuICogdGYubXVsdGlub21pYWwocHJvYnMsIDMpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gbG9naXRzIDFEIGFycmF5IHdpdGggdW5ub3JtYWxpemVkIGxvZy1wcm9iYWJpbGl0aWVzLCBvclxuICogICAgIDJEIGFycmF5IG9mIHNoYXBlIGBbYmF0Y2hTaXplLCBudW1PdXRjb21lc11gLiBTZWUgdGhlIGBub3JtYWxpemVkYFxuICogICAgIHBhcmFtZXRlci5cbiAqIEBwYXJhbSBudW1TYW1wbGVzIE51bWJlciBvZiBzYW1wbGVzIHRvIGRyYXcgZm9yIGVhY2ggcm93IHNsaWNlLlxuICogQHBhcmFtIHNlZWQgVGhlIHNlZWQgbnVtYmVyLlxuICogQHBhcmFtIG5vcm1hbGl6ZWQgV2hldGhlciB0aGUgcHJvdmlkZWQgYGxvZ2l0c2AgYXJlIG5vcm1hbGl6ZWQgdHJ1ZVxuICogICAgIHByb2JhYmlsaXRpZXMgKHN1bSB0byAxKS4gRGVmYXVsdHMgdG8gZmFsc2UuXG4gKiBAcmV0dXJuIDFEIGFycmF5IG9mIHNoYXBlIGBbbnVtU2FtcGxlc11gLCBvciAyRCBhcnJheSBvZiBzaGFwZVxuICogICAgIGBbYmF0Y2hTaXplLCBudW1TYW1wbGVzXWAsIGRlcGVuZGluZyBvbiB0aGUgcmFuayBvZiB0aGUgaW5wdXQuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnUmFuZG9tJ31cbiAqL1xuZnVuY3Rpb24gbXVsdGlub21pYWxfKFxuICAgIGxvZ2l0czogVGVuc29yMUR8VGVuc29yMkR8VGVuc29yTGlrZSwgbnVtU2FtcGxlczogbnVtYmVyLCBzZWVkPzogbnVtYmVyLFxuICAgIG5vcm1hbGl6ZWQgPSBmYWxzZSk6IFRlbnNvcjFEfFRlbnNvcjJEIHtcbiAgY29uc3QgJGxvZ2l0cyA9IGNvbnZlcnRUb1RlbnNvcihsb2dpdHMsICdsb2dpdHMnLCAnbXVsdGlub21pYWwnKTtcbiAgY29uc3QgbnVtT3V0Y29tZXMgPSAkbG9naXRzLnNpemU7XG4gIGNvbnN0IG9yaWdSYW5rID0gJGxvZ2l0cy5yYW5rO1xuICBpZiAobnVtT3V0Y29tZXMgPCAyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgRXJyb3IgaW4gbXVsdGlub21pYWw6IHlvdSBuZWVkIGF0IGxlYXN0IDIgb3V0Y29tZXMsIGJ1dCBnb3QgYCArXG4gICAgICAgIGAke251bU91dGNvbWVzfS5gKTtcbiAgfVxuICBpZiAob3JpZ1JhbmsgPiAyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBSYW5rIG9mIHByb2JhYmlsaXRpZXMgbXVzdCBiZSAxIG9yIDIsIGJ1dCBpcyAke29yaWdSYW5rfWApO1xuICB9XG4gIC8vIFRPRE8obGluYTEyOCk6IEludmVzdGlnYXRlIGNvcnJlY3Qgc2VlZCBiZWhhdmlvci4gVGhlIGNvZGUgc2VlbXMgbm90IGFsbG93XG4gIC8vIHNldHRpbmcgc2VlIHRvIDAuXG4gIHNlZWQgPSBzZWVkIHx8IE1hdGgucmFuZG9tKCk7XG5cbiAgLy8gVGhlIGtlcm5lbCBvbmx5IGFjY2VwdHMgKGFuZCByZXR1cm5zKSByYW5rIDIgdGVuc29ycy5cbiAgY29uc3QgbG9naXRzMkQ6IFRlbnNvcjJEID1cbiAgICAgIG9yaWdSYW5rID09PSAxID8gcmVzaGFwZSgkbG9naXRzLCBbMSwgLTFdKSA6ICRsb2dpdHMgYXMgVGVuc29yMkQ7XG5cbiAgY29uc3QgaW5wdXRzOiBNdWx0aW5vbWlhbElucHV0cyA9IHtsb2dpdHM6IGxvZ2l0czJEfTtcbiAgY29uc3QgYXR0cnM6IE11bHRpbm9taWFsQXR0cnMgPSB7bnVtU2FtcGxlcywgc2VlZCwgbm9ybWFsaXplZH07XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICBjb25zdCByZXMgPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICAgICAgTXVsdGlub21pYWwsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFRlbnNvcjJEO1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICByZXR1cm4gb3JpZ1JhbmsgPT09IDEgPyByZXNoYXBlKHJlcywgW3Jlcy5zaXplXSkgYXMgVGVuc29yMUQgOiByZXM7XG59XG5cbmV4cG9ydCBjb25zdCBtdWx0aW5vbWlhbCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bXVsdGlub21pYWxffSk7XG4iXX0=