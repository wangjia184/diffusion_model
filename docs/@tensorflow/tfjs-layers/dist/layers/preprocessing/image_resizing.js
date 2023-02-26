/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import { image, serialization, tidy } from '@tensorflow/tfjs-core'; // mul, add
import { Layer } from '../../engine/topology';
import { ValueError } from '../../errors';
import { getExactlyOneShape } from '../../utils/types_utils'; //, getExactlyOneTensor
// tf methods unimplemented in tfjs: 'bicubic', 'area', 'lanczos3', 'lanczos5',
//                                   'gaussian', 'mitchellcubic'
const INTERPOLATION_KEYS = ['bilinear', 'nearest'];
const INTERPOLATION_METHODS = new Set(INTERPOLATION_KEYS);
/**
 * Preprocessing Resizing Layer
 *
 * This resizes images by a scaling and offset factor
 */
export class Resizing extends Layer {
    constructor(args) {
        super(args);
        this.height = args.height;
        this.width = args.width;
        if (args.interpolation) {
            if (INTERPOLATION_METHODS.has(args.interpolation)) {
                this.interpolation = args.interpolation;
            }
            else {
                throw new ValueError(`Invalid interpolation parameter: ${args.interpolation} is not implemented`);
            }
        }
        else {
            this.interpolation = 'bilinear';
        }
        this.cropToAspectRatio = Boolean(args.cropToAspectRatio);
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const numChannels = inputShape[2];
        return [this.height, this.width, numChannels];
    }
    getConfig() {
        const config = {
            'height': this.height,
            'width': this.width,
            'interpolation': this.interpolation,
            'cropToAspectRatio': this.cropToAspectRatio
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            const size = [this.height, this.width];
            if (this.interpolation === 'bilinear') {
                return image.resizeBilinear(inputs, size, !this.cropToAspectRatio);
            }
            else if (this.interpolation === 'nearest') {
                return image.resizeNearestNeighbor(inputs, size, !this.cropToAspectRatio);
            }
            else {
                throw new Error(`Interpolation is ${this.interpolation} but only ${[...INTERPOLATION_METHODS]} are supported`);
            }
        });
    }
}
/** @nocollapse */
Resizing.className = 'Resizing';
serialization.registerClass(Resizing);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW1hZ2VfcmVzaXppbmcuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvbGF5ZXJzL3ByZXByb2Nlc3NpbmcvaW1hZ2VfcmVzaXppbmcudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSCxPQUFPLEVBQUMsS0FBSyxFQUFRLGFBQWEsRUFBVSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQyxDQUFFLFdBQVc7QUFFNUYsT0FBTyxFQUFDLEtBQUssRUFBWSxNQUFNLHVCQUF1QixDQUFDO0FBQ3ZELE9BQU8sRUFBQyxVQUFVLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFHeEMsT0FBTyxFQUFDLGtCQUFrQixFQUFDLE1BQU0seUJBQXlCLENBQUMsQ0FBRSx1QkFBdUI7QUFFcEYsK0VBQStFO0FBQy9FLGdFQUFnRTtBQUNoRSxNQUFNLGtCQUFrQixHQUFHLENBQUMsVUFBVSxFQUFFLFNBQVMsQ0FBVSxDQUFDO0FBQzVELE1BQU0scUJBQXFCLEdBQUcsSUFBSSxHQUFHLENBQUMsa0JBQWtCLENBQUMsQ0FBQztBQVUxRDs7OztHQUlHO0FBRUgsTUFBTSxPQUFPLFFBQVMsU0FBUSxLQUFLO0lBVWpDLFlBQVksSUFBa0I7UUFDNUIsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBRVosSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDO1FBQzFCLElBQUksQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQztRQUV4QixJQUFJLElBQUksQ0FBQyxhQUFhLEVBQUU7WUFDdEIsSUFBSSxxQkFBcUIsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxFQUFFO2dCQUNqRCxJQUFJLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUM7YUFDekM7aUJBQU07Z0JBQ0wsTUFBTSxJQUFJLFVBQVUsQ0FBQyxvQ0FDakIsSUFBSSxDQUFDLGFBQWEscUJBQXFCLENBQUMsQ0FBQzthQUM5QztTQUNGO2FBQU07WUFDTCxJQUFJLENBQUMsYUFBYSxHQUFHLFVBQVUsQ0FBQztTQUNqQztRQUNELElBQUksQ0FBQyxpQkFBaUIsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUM7SUFDM0QsQ0FBQztJQUVRLGtCQUFrQixDQUFDLFVBQXlCO1FBQ25ELFVBQVUsR0FBRyxrQkFBa0IsQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUM1QyxNQUFNLFdBQVcsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEMsT0FBTyxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsSUFBSSxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBRVEsU0FBUztRQUNoQixNQUFNLE1BQU0sR0FBNkI7WUFDdkMsUUFBUSxFQUFFLElBQUksQ0FBQyxNQUFNO1lBQ3JCLE9BQU8sRUFBRSxJQUFJLENBQUMsS0FBSztZQUNuQixlQUFlLEVBQUUsSUFBSSxDQUFDLGFBQWE7WUFDbkMsbUJBQW1CLEVBQUUsSUFBSSxDQUFDLGlCQUFpQjtTQUM1QyxDQUFDO1FBRUYsTUFBTSxVQUFVLEdBQUcsS0FBSyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ3JDLE1BQU0sQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLFVBQVUsQ0FBQyxDQUFDO1FBQ2xDLE9BQU8sTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFFUSxJQUFJLENBQUMsTUFBdUMsRUFBRSxNQUFjO1FBRW5FLE9BQU8sSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUNmLE1BQU0sSUFBSSxHQUFxQixDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQ3pELElBQUksSUFBSSxDQUFDLGFBQWEsS0FBSyxVQUFVLEVBQUU7Z0JBQ3JDLE9BQU8sS0FBSyxDQUFDLGNBQWMsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUM7YUFDcEU7aUJBQU0sSUFBSSxJQUFJLENBQUMsYUFBYSxLQUFLLFNBQVMsRUFBRTtnQkFDM0MsT0FBTyxLQUFLLENBQUMscUJBQXFCLENBQzlCLE1BQU0sRUFBRSxJQUFJLEVBQUUsQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQzthQUM1QztpQkFBTTtnQkFDTCxNQUFNLElBQUksS0FBSyxDQUFDLG9CQUFvQixJQUFJLENBQUMsYUFBYSxhQUFhLENBQUMsR0FBRyxxQkFBcUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDO2FBQ2hIO1FBQ0gsQ0FBQyxDQUFDLENBQUM7SUFDTCxDQUFDOztBQTVERCxrQkFBa0I7QUFDWCxrQkFBUyxHQUFHLFVBQVUsQ0FBQztBQThEaEMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxRQUFRLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIENvZGVTbWl0aCBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7aW1hZ2UsIFJhbmssIHNlcmlhbGl6YXRpb24sIFRlbnNvciwgdGlkeX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJzsgIC8vIG11bCwgYWRkXG5cbmltcG9ydCB7TGF5ZXIsIExheWVyQXJnc30gZnJvbSAnLi4vLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7VmFsdWVFcnJvcn0gZnJvbSAnLi4vLi4vZXJyb3JzJztcbmltcG9ydCB7U2hhcGV9IGZyb20gJy4uLy4uL2tlcmFzX2Zvcm1hdC9jb21tb24nO1xuaW1wb3J0IHtLd2FyZ3N9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCB7Z2V0RXhhY3RseU9uZVNoYXBlfSBmcm9tICcuLi8uLi91dGlscy90eXBlc191dGlscyc7ICAvLywgZ2V0RXhhY3RseU9uZVRlbnNvclxuXG4vLyB0ZiBtZXRob2RzIHVuaW1wbGVtZW50ZWQgaW4gdGZqczogJ2JpY3ViaWMnLCAnYXJlYScsICdsYW5jem9zMycsICdsYW5jem9zNScsXG4vLyAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2dhdXNzaWFuJywgJ21pdGNoZWxsY3ViaWMnXG5jb25zdCBJTlRFUlBPTEFUSU9OX0tFWVMgPSBbJ2JpbGluZWFyJywgJ25lYXJlc3QnXSBhcyBjb25zdDtcbmNvbnN0IElOVEVSUE9MQVRJT05fTUVUSE9EUyA9IG5ldyBTZXQoSU5URVJQT0xBVElPTl9LRVlTKTtcbnR5cGUgSW50ZXJwb2xhdGlvblR5cGUgPSB0eXBlb2YgSU5URVJQT0xBVElPTl9LRVlTW251bWJlcl07XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBSZXNpemluZ0FyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICBoZWlnaHQ6IG51bWJlcjtcbiAgd2lkdGg6IG51bWJlcjtcbiAgaW50ZXJwb2xhdGlvbj86IEludGVycG9sYXRpb25UeXBlOyAvLyBkZWZhdWx0ID0gJ2JpbGluZWFyJztcbiAgY3JvcFRvQXNwZWN0UmF0aW8/OiBib29sZWFuOyAgICAgICAvLyBkZWZhdWx0ID0gZmFsc2U7XG59XG5cbi8qKlxuICogUHJlcHJvY2Vzc2luZyBSZXNpemluZyBMYXllclxuICpcbiAqIFRoaXMgcmVzaXplcyBpbWFnZXMgYnkgYSBzY2FsaW5nIGFuZCBvZmZzZXQgZmFjdG9yXG4gKi9cblxuZXhwb3J0IGNsYXNzIFJlc2l6aW5nIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdSZXNpemluZyc7XG4gIHByaXZhdGUgcmVhZG9ubHkgaGVpZ2h0OiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgd2lkdGg6IG51bWJlcjtcbiAgLy8gbWV0aG9kIG9mIGludGVycG9sYXRpb24gdG8gYmUgdXNlZDsgZGVmYXVsdCA9IFwiYmlsaW5lYXJcIjtcbiAgcHJpdmF0ZSByZWFkb25seSBpbnRlcnBvbGF0aW9uOiBJbnRlcnBvbGF0aW9uVHlwZTtcbiAgLy8gdG9nZ2xlIHdoZXRoZXIgdGhlIGFzcGVjdCByYXRpbyBzaG91bGQgYmUgcHJlc2VydmVkOyBkZWZhdWx0ID0gZmFsc2U7XG4gIHByaXZhdGUgcmVhZG9ubHkgY3JvcFRvQXNwZWN0UmF0aW86IGJvb2xlYW47XG5cbiAgY29uc3RydWN0b3IoYXJnczogUmVzaXppbmdBcmdzKSB7XG4gICAgc3VwZXIoYXJncyk7XG5cbiAgICB0aGlzLmhlaWdodCA9IGFyZ3MuaGVpZ2h0O1xuICAgIHRoaXMud2lkdGggPSBhcmdzLndpZHRoO1xuXG4gICAgaWYgKGFyZ3MuaW50ZXJwb2xhdGlvbikge1xuICAgICAgaWYgKElOVEVSUE9MQVRJT05fTUVUSE9EUy5oYXMoYXJncy5pbnRlcnBvbGF0aW9uKSkge1xuICAgICAgICB0aGlzLmludGVycG9sYXRpb24gPSBhcmdzLmludGVycG9sYXRpb247XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihgSW52YWxpZCBpbnRlcnBvbGF0aW9uIHBhcmFtZXRlcjogJHtcbiAgICAgICAgICAgIGFyZ3MuaW50ZXJwb2xhdGlvbn0gaXMgbm90IGltcGxlbWVudGVkYCk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuaW50ZXJwb2xhdGlvbiA9ICdiaWxpbmVhcic7XG4gICAgfVxuICAgIHRoaXMuY3JvcFRvQXNwZWN0UmF0aW8gPSBCb29sZWFuKGFyZ3MuY3JvcFRvQXNwZWN0UmF0aW8pO1xuICB9XG5cbiAgb3ZlcnJpZGUgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuICAgIGNvbnN0IG51bUNoYW5uZWxzID0gaW5wdXRTaGFwZVsyXTtcbiAgICByZXR1cm4gW3RoaXMuaGVpZ2h0LCB0aGlzLndpZHRoLCBudW1DaGFubmVsc107XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgICdoZWlnaHQnOiB0aGlzLmhlaWdodCxcbiAgICAgICd3aWR0aCc6IHRoaXMud2lkdGgsXG4gICAgICAnaW50ZXJwb2xhdGlvbic6IHRoaXMuaW50ZXJwb2xhdGlvbixcbiAgICAgICdjcm9wVG9Bc3BlY3RSYXRpbyc6IHRoaXMuY3JvcFRvQXNwZWN0UmF0aW9cbiAgICB9O1xuXG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG5cbiAgb3ZlcnJpZGUgY2FsbChpbnB1dHM6IFRlbnNvcjxSYW5rLlIzPnxUZW5zb3I8UmFuay5SND4sIGt3YXJnczogS3dhcmdzKTpcbiAgICAgIFRlbnNvcltdfFRlbnNvciB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3Qgc2l6ZTogW251bWJlciwgbnVtYmVyXSA9IFt0aGlzLmhlaWdodCwgdGhpcy53aWR0aF07XG4gICAgICBpZiAodGhpcy5pbnRlcnBvbGF0aW9uID09PSAnYmlsaW5lYXInKSB7XG4gICAgICAgIHJldHVybiBpbWFnZS5yZXNpemVCaWxpbmVhcihpbnB1dHMsIHNpemUsICF0aGlzLmNyb3BUb0FzcGVjdFJhdGlvKTtcbiAgICAgIH0gZWxzZSBpZiAodGhpcy5pbnRlcnBvbGF0aW9uID09PSAnbmVhcmVzdCcpIHtcbiAgICAgICAgcmV0dXJuIGltYWdlLnJlc2l6ZU5lYXJlc3ROZWlnaGJvcihcbiAgICAgICAgICAgIGlucHV0cywgc2l6ZSwgIXRoaXMuY3JvcFRvQXNwZWN0UmF0aW8pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBJbnRlcnBvbGF0aW9uIGlzICR7dGhpcy5pbnRlcnBvbGF0aW9ufSBidXQgb25seSAke1suLi5JTlRFUlBPTEFUSU9OX01FVEhPRFNdfSBhcmUgc3VwcG9ydGVkYCk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cbn1cblxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFJlc2l6aW5nKTtcbiJdfQ==