import { convertToTensor } from '../tensor_util_env';
import { conv3DBackpropInput } from './conv3d_backprop_input';
import { op } from './operation';
/**
 * Computes the transposed 3D convolution of a volume, also known as a
 * deconvolution.
 *
 * @param x The input image, of rank 5 or rank 4, of shape
 *   `[batch, depth, height, width, inDepth]`. If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 4, of shape
 *     `[depth, filterHeight, filterWidth, outDepth, inDepth]`.
 *     `inDepth` must match `inDepth` in `x`.
 * @param outputShape Output shape, of rank 5 or rank 4:
 *     `[batch, depth, height, width, outDepth]`. If rank 3, batch of 1 is
 *    assumed.
 * @param strides The strides of the original convolution:
 *     `[strideDepth, strideHeight, strideWidth]`.
 * @param pad  The type of padding algorithm used in the non-transpose version
 *    of the op.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv3dTranspose_(x, filter, outputShape, strides, pad) {
    const $x = convertToTensor(x, 'x', 'conv3dTranspose');
    const $filter = convertToTensor(filter, 'filter', 'conv3dTranspose');
    return conv3DBackpropInput(outputShape, $x, $filter, strides, pad);
}
export const conv3dTranspose = /* @__PURE__ */ op({ conv3dTranspose_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjNkX3RyYW5zcG9zZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NvbnYzZF90cmFuc3Bvc2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBaUJBLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FrQkc7QUFDSCxTQUFTLGdCQUFnQixDQUNyQixDQUFlLEVBQUUsTUFBMkIsRUFDNUMsV0FFNkMsRUFDN0MsT0FBd0MsRUFBRSxHQUFtQjtJQUMvRCxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDO0lBQ3RELE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLGlCQUFpQixDQUFDLENBQUM7SUFFckUsT0FBTyxtQkFBbUIsQ0FBQyxXQUFXLEVBQUUsRUFBRSxFQUFFLE9BQU8sRUFBRSxPQUFPLEVBQUUsR0FBRyxDQUFDLENBQUM7QUFDckUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGVBQWUsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsZ0JBQWdCLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtUZW5zb3I0RCwgVGVuc29yNUR9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge2NvbnYzREJhY2twcm9wSW5wdXR9IGZyb20gJy4vY29udjNkX2JhY2twcm9wX2lucHV0JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgdHJhbnNwb3NlZCAzRCBjb252b2x1dGlvbiBvZiBhIHZvbHVtZSwgYWxzbyBrbm93biBhcyBhXG4gKiBkZWNvbnZvbHV0aW9uLlxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgb2YgcmFuayA1IG9yIHJhbmsgNCwgb2Ygc2hhcGVcbiAqICAgYFtiYXRjaCwgZGVwdGgsIGhlaWdodCwgd2lkdGgsIGluRGVwdGhdYC4gSWYgcmFuayA0LCBiYXRjaCBvZiAxIGlzIGFzc3VtZWQuXG4gKiBAcGFyYW0gZmlsdGVyIFRoZSBmaWx0ZXIsIHJhbmsgNCwgb2Ygc2hhcGVcbiAqICAgICBgW2RlcHRoLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBvdXREZXB0aCwgaW5EZXB0aF1gLlxuICogICAgIGBpbkRlcHRoYCBtdXN0IG1hdGNoIGBpbkRlcHRoYCBpbiBgeGAuXG4gKiBAcGFyYW0gb3V0cHV0U2hhcGUgT3V0cHV0IHNoYXBlLCBvZiByYW5rIDUgb3IgcmFuayA0OlxuICogICAgIGBbYmF0Y2gsIGRlcHRoLCBoZWlnaHQsIHdpZHRoLCBvdXREZXB0aF1gLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXNcbiAqICAgIGFzc3VtZWQuXG4gKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgb3JpZ2luYWwgY29udm9sdXRpb246XG4gKiAgICAgYFtzdHJpZGVEZXB0aCwgc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aF1gLlxuICogQHBhcmFtIHBhZCAgVGhlIHR5cGUgb2YgcGFkZGluZyBhbGdvcml0aG0gdXNlZCBpbiB0aGUgbm9uLXRyYW5zcG9zZSB2ZXJzaW9uXG4gKiAgICBvZiB0aGUgb3AuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb24nfVxuICovXG5mdW5jdGlvbiBjb252M2RUcmFuc3Bvc2VfPFQgZXh0ZW5kcyBUZW5zb3I0RHxUZW5zb3I1RD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBmaWx0ZXI6IFRlbnNvcjVEfFRlbnNvckxpa2UsXG4gICAgb3V0cHV0U2hhcGU6XG4gICAgICAgIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsXG4gICAgICAgICBudW1iZXJdfFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXXxudW1iZXIsIHBhZDogJ3ZhbGlkJ3wnc2FtZScpOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnY29udjNkVHJhbnNwb3NlJyk7XG4gIGNvbnN0ICRmaWx0ZXIgPSBjb252ZXJ0VG9UZW5zb3IoZmlsdGVyLCAnZmlsdGVyJywgJ2NvbnYzZFRyYW5zcG9zZScpO1xuXG4gIHJldHVybiBjb252M0RCYWNrcHJvcElucHV0KG91dHB1dFNoYXBlLCAkeCwgJGZpbHRlciwgc3RyaWRlcywgcGFkKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNvbnYzZFRyYW5zcG9zZSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7Y29udjNkVHJhbnNwb3NlX30pO1xuIl19