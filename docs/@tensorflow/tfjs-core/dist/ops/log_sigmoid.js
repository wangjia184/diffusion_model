/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { customGrad } from '../gradients';
import { convertToTensor } from '../tensor_util_env';
import { mul } from './mul';
import { neg } from './neg';
import { op } from './operation';
import { sigmoid } from './sigmoid';
import { softplus } from './softplus';
/**
 * Computes log sigmoid of the input `tf.Tensor` element-wise:
 * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.logSigmoid().print();  // or tf.logSigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function logSigmoid_(x) {
    const $x = convertToTensor(x, 'x', 'logSigmoid');
    // Use a custom gradient to maintain previous implementation.
    // There is no LogSigmoid kernel in TF so we can't use engine.runKernel
    // directly
    const customOp = customGrad((x) => {
        // TODO(yassogba) we can remove the chained softplus call here only
        // after backends have modualrized softplus at which point we can call
        // engine runKernel(..., Sotfplus, ...) directly.
        const value = neg(softplus(neg(x)));
        const gradFunc = (dy) => {
            const derX = mul(dy, sigmoid(neg(x)));
            return derX;
        };
        return { value, gradFunc };
    });
    return customOp($x);
}
export const logSigmoid = /* @__PURE__ */ op({ logSigmoid_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9nX3NpZ21vaWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9sb2dfc2lnbW9pZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRXhDLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFFcEM7Ozs7Ozs7Ozs7OztHQVlHO0FBQ0gsU0FBUyxXQUFXLENBQW1CLENBQWU7SUFDcEQsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsWUFBWSxDQUFDLENBQUM7SUFFakQsNkRBQTZEO0lBQzdELHVFQUF1RTtJQUN2RSxXQUFXO0lBQ1gsTUFBTSxRQUFRLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBUyxFQUFFLEVBQUU7UUFDeEMsbUVBQW1FO1FBQ25FLHNFQUFzRTtRQUN0RSxpREFBaUQ7UUFDakQsTUFBTSxLQUFLLEdBQUcsR0FBRyxDQUFDLFFBQVEsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRXBDLE1BQU0sUUFBUSxHQUFHLENBQUMsRUFBSyxFQUFFLEVBQUU7WUFDekIsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDLEVBQUUsRUFBRSxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN0QyxPQUFPLElBQUksQ0FBQztRQUNkLENBQUMsQ0FBQztRQUNGLE9BQU8sRUFBQyxLQUFLLEVBQUUsUUFBUSxFQUFDLENBQUM7SUFDM0IsQ0FBQyxDQUFDLENBQUM7SUFFSCxPQUFPLFFBQVEsQ0FBQyxFQUFFLENBQU0sQ0FBQztBQUMzQixDQUFDO0FBQ0QsTUFBTSxDQUFDLE1BQU0sVUFBVSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxXQUFXLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2N1c3RvbUdyYWR9IGZyb20gJy4uL2dyYWRpZW50cyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7bXVsfSBmcm9tICcuL211bCc7XG5pbXBvcnQge25lZ30gZnJvbSAnLi9uZWcnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtzaWdtb2lkfSBmcm9tICcuL3NpZ21vaWQnO1xuaW1wb3J0IHtzb2Z0cGx1c30gZnJvbSAnLi9zb2Z0cGx1cyc7XG5cbi8qKlxuICogQ29tcHV0ZXMgbG9nIHNpZ21vaWQgb2YgdGhlIGlucHV0IGB0Zi5UZW5zb3JgIGVsZW1lbnQtd2lzZTpcbiAqIGBsb2dTaWdtb2lkKHgpYC4gRm9yIG51bWVyaWNhbCBzdGFiaWxpdHksIHdlIHVzZSBgLXRmLnNvZnRwbHVzKC14KWAuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMCwgMSwgLTEsIC43XSk7XG4gKlxuICogeC5sb2dTaWdtb2lkKCkucHJpbnQoKTsgIC8vIG9yIHRmLmxvZ1NpZ21vaWQoeClcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvci5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdCYXNpYyBtYXRoJ31cbiAqL1xuZnVuY3Rpb24gbG9nU2lnbW9pZF88VCBleHRlbmRzIFRlbnNvcj4oeDogVHxUZW5zb3JMaWtlKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2xvZ1NpZ21vaWQnKTtcblxuICAvLyBVc2UgYSBjdXN0b20gZ3JhZGllbnQgdG8gbWFpbnRhaW4gcHJldmlvdXMgaW1wbGVtZW50YXRpb24uXG4gIC8vIFRoZXJlIGlzIG5vIExvZ1NpZ21vaWQga2VybmVsIGluIFRGIHNvIHdlIGNhbid0IHVzZSBlbmdpbmUucnVuS2VybmVsXG4gIC8vIGRpcmVjdGx5XG4gIGNvbnN0IGN1c3RvbU9wID0gY3VzdG9tR3JhZCgoeDogVGVuc29yKSA9PiB7XG4gICAgLy8gVE9ETyh5YXNzb2diYSkgd2UgY2FuIHJlbW92ZSB0aGUgY2hhaW5lZCBzb2Z0cGx1cyBjYWxsIGhlcmUgb25seVxuICAgIC8vIGFmdGVyIGJhY2tlbmRzIGhhdmUgbW9kdWFscml6ZWQgc29mdHBsdXMgYXQgd2hpY2ggcG9pbnQgd2UgY2FuIGNhbGxcbiAgICAvLyBlbmdpbmUgcnVuS2VybmVsKC4uLiwgU290ZnBsdXMsIC4uLikgZGlyZWN0bHkuXG4gICAgY29uc3QgdmFsdWUgPSBuZWcoc29mdHBsdXMobmVnKHgpKSk7XG5cbiAgICBjb25zdCBncmFkRnVuYyA9IChkeTogVCkgPT4ge1xuICAgICAgY29uc3QgZGVyWCA9IG11bChkeSwgc2lnbW9pZChuZWcoeCkpKTtcbiAgICAgIHJldHVybiBkZXJYO1xuICAgIH07XG4gICAgcmV0dXJuIHt2YWx1ZSwgZ3JhZEZ1bmN9O1xuICB9KTtcblxuICByZXR1cm4gY3VzdG9tT3AoJHgpIGFzIFQ7XG59XG5leHBvcnQgY29uc3QgbG9nU2lnbW9pZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bG9nU2lnbW9pZF99KTtcbiJdfQ==