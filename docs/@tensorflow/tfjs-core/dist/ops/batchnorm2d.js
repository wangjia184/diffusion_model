import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { batchNorm } from './batchnorm';
import { op } from './operation';
/**
 * Batch normalization, strictly for 2D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm2d_(x, mean, variance, offset, scale, varianceEpsilon) {
    const $x = convertToTensor(x, 'x', 'batchNorm');
    const $mean = convertToTensor(mean, 'mean', 'batchNorm');
    const $variance = convertToTensor(variance, 'variance', 'batchNorm');
    let $scale;
    if (scale != null) {
        $scale = convertToTensor(scale, 'scale', 'batchNorm');
    }
    let $offset;
    if (offset != null) {
        $offset = convertToTensor(offset, 'offset', 'batchNorm');
    }
    util.assert($x.rank === 2, () => `Error in batchNorm2D: x must be rank 2 but got rank ` +
        `${$x.rank}.`);
    util.assert($mean.rank === 2 || $mean.rank === 1, () => `Error in batchNorm2D: mean must be rank 2 or rank 1 but ` +
        `got rank ${$mean.rank}.`);
    util.assert($variance.rank === 2 || $variance.rank === 1, () => `Error in batchNorm2D: variance must be rank 2 or rank 1 ` +
        `but got rank ${$variance.rank}.`);
    if ($scale != null) {
        util.assert($scale.rank === 2 || $scale.rank === 1, () => `Error in batchNorm2D: scale must be rank 2 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
    }
    if ($offset != null) {
        util.assert($offset.rank === 2 || $offset.rank === 1, () => `Error in batchNorm2D: offset must be rank 2 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
    }
    return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
export const batchNorm2d = /* @__PURE__ */ op({ batchNorm2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmF0Y2hub3JtMmQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9iYXRjaG5vcm0yZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFpQkEsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDdEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7OztHQVVHO0FBQ0gsU0FBUyxZQUFZLENBQ2pCLENBQXNCLEVBQUUsSUFBa0MsRUFDMUQsUUFBc0MsRUFDdEMsTUFBcUMsRUFBRSxLQUFvQyxFQUMzRSxlQUF3QjtJQUMxQixNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNoRCxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQztJQUN6RCxNQUFNLFNBQVMsR0FBRyxlQUFlLENBQUMsUUFBUSxFQUFFLFVBQVUsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNyRSxJQUFJLE1BQXlCLENBQUM7SUFDOUIsSUFBSSxLQUFLLElBQUksSUFBSSxFQUFFO1FBQ2pCLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxXQUFXLENBQUMsQ0FBQztLQUN2RDtJQUNELElBQUksT0FBMEIsQ0FBQztJQUMvQixJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7UUFDbEIsT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0tBQzFEO0lBQ0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDYixHQUFHLEVBQUUsQ0FBQyxzREFBc0Q7UUFDeEQsR0FBRyxFQUFFLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUN2QixJQUFJLENBQUMsTUFBTSxDQUNQLEtBQUssQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNwQyxHQUFHLEVBQUUsQ0FBQywwREFBMEQ7UUFDNUQsWUFBWSxLQUFLLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNuQyxJQUFJLENBQUMsTUFBTSxDQUNQLFNBQVMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLFNBQVMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUM1QyxHQUFHLEVBQUUsQ0FBQywwREFBMEQ7UUFDNUQsZ0JBQWdCLFNBQVMsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQzNDLElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtRQUNsQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUN0QyxHQUFHLEVBQUUsQ0FBQyx1REFBdUQ7WUFDekQsZ0JBQWdCLE1BQU0sQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0tBQ3pDO0lBQ0QsSUFBSSxPQUFPLElBQUksSUFBSSxFQUFFO1FBQ25CLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ3hDLEdBQUcsRUFBRSxDQUFDLHdEQUF3RDtZQUMxRCxnQkFBZ0IsT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7S0FDMUM7SUFFRCxPQUFPLFNBQVMsQ0FBQyxFQUFFLEVBQUUsS0FBSyxFQUFFLFNBQVMsRUFBRSxPQUFPLEVBQUUsTUFBTSxFQUFFLGVBQWUsQ0FBQyxDQUFDO0FBQzNFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge1RlbnNvcjFELCBUZW5zb3IyRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2JhdGNoTm9ybX0gZnJvbSAnLi9iYXRjaG5vcm0nO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEJhdGNoIG5vcm1hbGl6YXRpb24sIHN0cmljdGx5IGZvciAyRC4gRm9yIHRoZSBtb3JlIHJlbGF4ZWQgdmVyc2lvbiwgc2VlXG4gKiBgdGYuYmF0Y2hOb3JtYC5cbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgVGVuc29yLlxuICogQHBhcmFtIG1lYW4gQSBtZWFuIFRlbnNvci5cbiAqIEBwYXJhbSB2YXJpYW5jZSBBIHZhcmlhbmNlIFRlbnNvci5cbiAqIEBwYXJhbSBvZmZzZXQgQW4gb2Zmc2V0IFRlbnNvci5cbiAqIEBwYXJhbSBzY2FsZSBBIHNjYWxlIFRlbnNvci5cbiAqIEBwYXJhbSB2YXJpYW5jZUVwc2lsb24gQSBzbWFsbCBmbG9hdCBudW1iZXIgdG8gYXZvaWQgZGl2aWRpbmcgYnkgMC5cbiAqL1xuZnVuY3Rpb24gYmF0Y2hOb3JtMmRfKFxuICAgIHg6IFRlbnNvcjJEfFRlbnNvckxpa2UsIG1lYW46IFRlbnNvcjJEfFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgdmFyaWFuY2U6IFRlbnNvcjJEfFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgb2Zmc2V0PzogVGVuc29yMkR8VGVuc29yMUR8VGVuc29yTGlrZSwgc2NhbGU/OiBUZW5zb3IyRHxUZW5zb3IxRHxUZW5zb3JMaWtlLFxuICAgIHZhcmlhbmNlRXBzaWxvbj86IG51bWJlcik6IFRlbnNvcjJEIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnYmF0Y2hOb3JtJyk7XG4gIGNvbnN0ICRtZWFuID0gY29udmVydFRvVGVuc29yKG1lYW4sICdtZWFuJywgJ2JhdGNoTm9ybScpO1xuICBjb25zdCAkdmFyaWFuY2UgPSBjb252ZXJ0VG9UZW5zb3IodmFyaWFuY2UsICd2YXJpYW5jZScsICdiYXRjaE5vcm0nKTtcbiAgbGV0ICRzY2FsZTogVGVuc29yMkR8VGVuc29yMUQ7XG4gIGlmIChzY2FsZSAhPSBudWxsKSB7XG4gICAgJHNjYWxlID0gY29udmVydFRvVGVuc29yKHNjYWxlLCAnc2NhbGUnLCAnYmF0Y2hOb3JtJyk7XG4gIH1cbiAgbGV0ICRvZmZzZXQ6IFRlbnNvcjJEfFRlbnNvcjFEO1xuICBpZiAob2Zmc2V0ICE9IG51bGwpIHtcbiAgICAkb2Zmc2V0ID0gY29udmVydFRvVGVuc29yKG9mZnNldCwgJ29mZnNldCcsICdiYXRjaE5vcm0nKTtcbiAgfVxuICB1dGlsLmFzc2VydChcbiAgICAgICR4LnJhbmsgPT09IDIsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gYmF0Y2hOb3JtMkQ6IHggbXVzdCBiZSByYW5rIDIgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgIGAkeyR4LnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRtZWFuLnJhbmsgPT09IDIgfHwgJG1lYW4ucmFuayA9PT0gMSxcbiAgICAgICgpID0+IGBFcnJvciBpbiBiYXRjaE5vcm0yRDogbWVhbiBtdXN0IGJlIHJhbmsgMiBvciByYW5rIDEgYnV0IGAgK1xuICAgICAgICAgIGBnb3QgcmFuayAkeyRtZWFuLnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICR2YXJpYW5jZS5yYW5rID09PSAyIHx8ICR2YXJpYW5jZS5yYW5rID09PSAxLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGJhdGNoTm9ybTJEOiB2YXJpYW5jZSBtdXN0IGJlIHJhbmsgMiBvciByYW5rIDEgYCArXG4gICAgICAgICAgYGJ1dCBnb3QgcmFuayAkeyR2YXJpYW5jZS5yYW5rfS5gKTtcbiAgaWYgKCRzY2FsZSAhPSBudWxsKSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICRzY2FsZS5yYW5rID09PSAyIHx8ICRzY2FsZS5yYW5rID09PSAxLFxuICAgICAgICAoKSA9PiBgRXJyb3IgaW4gYmF0Y2hOb3JtMkQ6IHNjYWxlIG11c3QgYmUgcmFuayAyIG9yIHJhbmsgMSBgICtcbiAgICAgICAgICAgIGBidXQgZ290IHJhbmsgJHskc2NhbGUucmFua30uYCk7XG4gIH1cbiAgaWYgKCRvZmZzZXQgIT0gbnVsbCkge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAkb2Zmc2V0LnJhbmsgPT09IDIgfHwgJG9mZnNldC5yYW5rID09PSAxLFxuICAgICAgICAoKSA9PiBgRXJyb3IgaW4gYmF0Y2hOb3JtMkQ6IG9mZnNldCBtdXN0IGJlIHJhbmsgMiBvciByYW5rIDEgYCArXG4gICAgICAgICAgICBgYnV0IGdvdCByYW5rICR7JG9mZnNldC5yYW5rfS5gKTtcbiAgfVxuXG4gIHJldHVybiBiYXRjaE5vcm0oJHgsICRtZWFuLCAkdmFyaWFuY2UsICRvZmZzZXQsICRzY2FsZSwgdmFyaWFuY2VFcHNpbG9uKTtcbn1cblxuZXhwb3J0IGNvbnN0IGJhdGNoTm9ybTJkID0gLyogQF9fUFVSRV9fICovIG9wKHtiYXRjaE5vcm0yZF99KTtcbiJdfQ==