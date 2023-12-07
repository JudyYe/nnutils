import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable
import torch
from pytorch3d.structures import Meshes as MeshesBase
from pytorch3d.structures import Volumes as VolumeBase
from pytorch3d.loss.chamfer import _handle_pointcloud_input, knn_gather, knn_points
from pytorch3d.renderer import MultinomialRaysampler
from pytorch3d.renderer.implicit.raysampling import (
    _jiggle_within_stratas, _sample_cameras_and_masks, _pack_ray_bundle, _safe_multinomial,
    RayBundle, HeterogeneousRayBundle, CamerasBase)
# class Volumes(VolumeBase):
#     def __init__(self, densities: _TensorBatch, features: _TensorBatch | None = None, voxel_size: _VoxelSize = 1, volume_translation: _Translation = ...) -> None:
#         super().__init__(densities, features, voxel_size, volume_translation)
#         self.voxel_size = voxel_size
#         self.translation = volume_translation
# # overwrite pytorch3d: Meshes



class BatchMultinomialRaysampler(MultinomialRaysampler):
    """
    A batched version of MultinomialRaysampler, supports batched min_depth
    """

    def __init__(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
        """
        Args:
            min_x: The leftmost x-coordinate of each ray's source pixel's center.
            max_x: The rightmost x-coordinate of each ray's source pixel's center.
            min_y: The topmost y-coordinate of each ray's source pixel's center.
            max_y: The bottommost y-coordinate of each ray's source pixel's center.
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: If given, this amount of rays are sampled from the grid.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            unit_directions: whether to normalize direction vectors in ray bundle.
            stratified_sampling: if True, performs stratified random sampling
                along the ray; otherwise takes ray points at deterministic offsets.
        """
        super().__init__(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                         image_width=image_width, image_height=image_height,
                         n_pts_per_ray=n_pts_per_ray, min_depth=min_depth,
                         max_depth=max_depth, n_rays_per_image=n_rays_per_image,
                         n_rays_total=n_rays_total, unit_directions=unit_directions,
                         stratified_sampling=stratified_sampling)
        

    def forward(
        self,
        cameras: CamerasBase,
        *,
        mask: Optional[torch.Tensor] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        n_rays_per_image: Optional[int] = None,
        n_pts_per_ray: Optional[int] = None,
        stratified_sampling: Optional[bool] = None,
        n_rays_total: Optional[int] = None,
        **kwargs,
    ) -> Union[RayBundle, HeterogeneousRayBundle]:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            mask: if given, the rays are sampled from the mask. Should be of size
                (batch_size, image_height, image_width).
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: If given, this amount of rays are sampled from the grid.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
            n_pts_per_ray: The number of points sampled along each ray.
            stratified_sampling: if set, overrides stratified_sampling provided
                in __init__.
            n_rays_total: How many rays in total to sample from the cameras provided. The result
                is as if `n_rays_total_training` cameras were sampled with replacement from the
                cameras provided and for every camera one ray was sampled. If set returns the
                HeterogeneousRayBundle with batch_size=n_rays_total.
                `n_rays_per_image` and `n_rays_total` cannot both be defined.
        Returns:
            A named tuple RayBundle or dataclass HeterogeneousRayBundle with the
            following fields:

            origins: A tensor of shape
                `(batch_size, s1, s2, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, s1, s2, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, s1, s2, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, s1, s2, 2)`
                containing the 2D image coordinates of each ray or,
                if mask is given, `(batch_size, n, 1, 2)`
            Here `s1, s2` refer to spatial dimensions.
            `(s1, s2)` refer to (highest priority first):
                - `(1, 1)` if `n_rays_total` is provided, (batch_size=n_rays_total)
                - `(n_rays_per_image, 1) if `n_rays_per_image` if provided,
                - `(n, 1)` where n is the minimum cardinality of the mask
                        in the batch if `mask` is provided
                - `(image_height, image_width)` if nothing from above is satisfied

            `HeterogeneousRayBundle` has additional members:
                - camera_ids: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. It represents unique ids of sampled cameras.
                - camera_counts: tensor of shape (M,), where `M` is the number of unique sampled
                    cameras. Represents how many times each camera from `camera_ids` was sampled

            `HeterogeneousRayBundle` is returned if `n_rays_total` is provided else `RayBundle`
            is returned.
        """
        n_rays_total = n_rays_total or self._n_rays_total
        n_rays_per_image = n_rays_per_image or self._n_rays_per_image
        if (n_rays_total is not None) and (n_rays_per_image is not None):
            raise ValueError(
                "`n_rays_total` and `n_rays_per_image` cannot both be defined."
            )
        if n_rays_total:
            (
                cameras,
                mask,
                camera_ids,  # unique ids of sampled cameras
                camera_counts,  # number of times unique camera id was sampled
                # `n_rays_per_image` is equal to the max number of times a simgle camera
                # was sampled. We sample all cameras at `camera_ids` `n_rays_per_image` times
                # and then discard the unneeded rays.
                # pyre-ignore[9]
                n_rays_per_image,
            ) = _sample_cameras_and_masks(n_rays_total, cameras, mask)
        else:
            # pyre-ignore[9]
            camera_ids: torch.LongTensor = torch.arange(len(cameras), dtype=torch.long)

        batch_size = cameras.R.shape[0]
        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device).expand(batch_size, -1, -1, -1)

        if mask is not None and n_rays_per_image is None:
            # if num rays not given, sample according to the smallest mask
            n_rays_per_image = (
                n_rays_per_image or mask.sum(dim=(1, 2)).min().int().item()
            )

        if n_rays_per_image is not None:
            if mask is not None:
                assert mask.shape == xy_grid.shape[:3]
                weights = mask.reshape(batch_size, -1)
            else:
                # it is probably more efficient to use torch.randperm
                # for uniform weights but it is unlikely given that randperm
                # is not batched and does not support partial permutation
                _, width, height, _ = xy_grid.shape
                weights = xy_grid.new_ones(batch_size, width * height)
            # pyre-fixme[6]: For 2nd param expected `int` but got `Union[bool,
            #  float, int]`.
            rays_idx = _safe_multinomial(weights, n_rays_per_image)[..., None].expand(
                -1, -1, 2
            )

            xy_grid = torch.gather(xy_grid.reshape(batch_size, -1, 2), 1, rays_idx)[
                :, :, None
            ]

        min_depth = min_depth if min_depth is not None else self._min_depth
        max_depth = max_depth if max_depth is not None else self._max_depth
        n_pts_per_ray = (
            n_pts_per_ray if n_pts_per_ray is not None else self._n_pts_per_ray
        )
        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        ray_bundle = _xy_to_ray_bundle(
            cameras,
            xy_grid,
            min_depth,
            max_depth,
            n_pts_per_ray,
            self._unit_directions,
            stratified_sampling,
        )

        return (
            # pyre-ignore[61]
            _pack_ray_bundle(ray_bundle, camera_ids, camera_counts)
            if n_rays_total
            else ray_bundle
        )


def _xy_to_ray_bundle(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    min_depth: float,
    max_depth: float,
    n_pts_per_ray: int,
    unit_directions: bool,
    stratified_sampling: bool = False,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.

    Args:
        cameras: cameras object representing a batch of cameras.
        xy_grid: torch.tensor grid of image xy coords.
        min_depth: The minimum depth of each ray-point.
        max_depth: The maximum depth of each ray-point.
        n_pts_per_ray: The number of points sampled along each ray.
        unit_directions: whether to normalize direction vectors in ray bundle.
        stratified_sampling: if True, performs stratified sampling in n_pts_per_ray
            bins for each ray; otherwise takes n_pts_per_ray deterministic points
            on each ray with uniform offsets.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()

    # ray z-coords
    rays_zs = xy_grid.new_empty((0,))
    if n_pts_per_ray > 0:
        # CMT: changed
        if not torch.is_tensor(min_depth):
            min_depth = torch.zeros([batch_size, 1], device=xy_grid.device) + min_depth
        if not torch.is_tensor(max_depth):
            max_depth = torch.zeros([batch_size, 1], device=xy_grid.device) + max_depth
        min_depth = min_depth.view(batch_size, 1)  # (N, 1)
        max_depth = max_depth.view(batch_size, 1)  # (N, 1)
        depths = torch.linspace(0, 1, n_pts_per_ray, dtype=xy_grid.dtype, device=xy_grid.device)[None]  # (1, D)
        depths = depths * (max_depth - min_depth) + min_depth  # (N, D)
        rays_zs = depths.unsqueeze(1).expand(batch_size, n_rays_per_image, n_pts_per_ray)

        # depths = torch.linspace(
        #     min_depth,
        #     max_depth,
        #     n_pts_per_ray,
        #     dtype=xy_grid.dtype,
        #     device=xy_grid.device,
        # )
        # rays_zs = depths.unsqueeze(1).expand(batch_size, n_rays_per_image, n_pts_per_ray)


        if stratified_sampling:
            rays_zs = _jiggle_within_stratas(rays_zs)

    # make two sets of points at a constant depth=1 and 2
    to_unproject = torch.cat(
        (
            xy_grid.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject, from_ndc=True)

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    if unit_directions:
        rays_directions_world = F.normalize(rays_directions_world, dim=-1)

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )



class BatchedNDCMultinomialRaysampler(BatchMultinomialRaysampler):

    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        n_rays_total: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
        if image_width >= image_height:
            range_x = image_width / image_height
            range_y = 1.0
        else:
            range_x = 1.0
            range_y = image_height / image_width

        half_pix_width = range_x / image_width
        half_pix_height = range_y / image_height
        super().__init__(
            min_x=range_x - half_pix_width,
            max_x=-range_x + half_pix_width,
            min_y=range_y - half_pix_height,
            max_y=-range_y + half_pix_height,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            n_rays_per_image=n_rays_per_image,
            n_rays_total=n_rays_total,
            unit_directions=unit_directions,
            stratified_sampling=stratified_sampling,
        )


class Meshes(MeshesBase):

    def __init__(
        self,
        verts=None,
        faces=None,
        textures=None,
        *,
        verts_normals=None,
    ) -> None:
        """
        Args:
            verts:
                Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the (x, y, z) coordinates of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of vertices.
            faces:
                Can be either

                - List where each element is a tensor of shape (num_faces, 3)
                  containing the indices of the 3 vertices in the corresponding
                  mesh in verts which form the triangular face.
                - Padded long tensor of shape (num_meshes, max_num_faces, 3).
                  Meshes should be padded with fill value of -1 so they have
                  the same number of faces.
            textures: Optional instance of the Textures class with mesh
                texture properties.
            verts_normals:
                Optional. Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the normals of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  They should be padded with fill value of 0 so they all have
                  the same number of vertices.
                Note that modifying the mesh later, e.g. with offset_verts_,
                can cause these normals to be forgotten and normals to be recalculated
                based on the new vertex positions.

        Refer to comments above for descriptions of List and Padded representations.
        """
        super().__init__(verts, faces, textures, verts_normals=verts_normals)
        # reset shape for empty meshes
        if self.isempty():

            # Identify type of verts and faces.
            if isinstance(verts, list) and isinstance(faces, list):
                if self._N > 0:
                    # if not (
                    #     all(v.device == self.device for v in verts)
                    #     and all(f.device == self.device for f in faces)
                    # ):
                    self._num_verts_per_mesh = torch.tensor(
                        [len(v) for v in self._verts_list], device=self.device
                    )
                    self._num_faces_per_mesh = torch.tensor(
                        [len(f) for f in self._faces_list], device=self.device
                    )
            elif torch.is_tensor(verts) and torch.is_tensor(faces):
                if self._N > 0:
                    # Check that padded faces - which have value -1 - are at the
                    # end of the tensors
                    faces_not_padded = self._faces_padded.gt(-1).all(2)
                    self._num_faces_per_mesh = faces_not_padded.sum(1)

                    self._num_verts_per_mesh = torch.full(
                        size=(self._N,),
                        fill_value=self._V,
                        dtype=torch.int64,
                        device=self.device,
                    )

            else:
                raise ValueError(
                    "Verts and Faces must be either a list or a tensor with \
                        shape (batch_size, N, 3) where N is either the maximum \
                        number of verts or faces respectively."
                )

            # if self.isempty():
            #     self._num_verts_per_mesh = torch.zeros(
            #         (0,), dtype=torch.int64, device=self.device
            #     )
            #     self._num_faces_per_mesh = torch.zeros(
            #         (0,), dtype=torch.int64, device=self.device
            #     )

            # Set the num verts/faces on the textures if present.
            if textures is not None:
                shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
                if not shape_ok:
                    msg = "Textures do not match the dimensions of Meshes."
                    raise ValueError(msg)

                self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
                self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()
                self.textures.valid = self.valid

            if verts_normals is not None:
                self._set_verts_normals(verts_normals)
        else:
            pass





# overwrite pytorch3d: chamfer distance
def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    if point_reduction is None:
        return (cham_x, cham_y), (cham_norm_x, cham_norm_y)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals

from pytorch3d.renderer.mesh.shader import ShaderBase, hard_rgb_blend

# double side shader
class DoulbeSideHardPhongShader(ShaderBase):

    def forward(self, fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading_2side(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images



def phong_shading_2side(
    meshes, fragments, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    colors, _ = _phong_shading_with_pixels_2side(
        meshes, fragments, lights, cameras, materials, texels
    )
    return colors

from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.shading import _apply_lighting


def _phong_shading_with_pixels_2side(
    meshes, fragments, lights, cameras, materials, texels
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
        pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords_in_camera, pixel_normals, lights, cameras, materials
    )
    _, diffuse_rev, _ = _apply_lighting(
        pixel_coords_in_camera, -pixel_normals, lights, cameras, materials
    )
    diffuse = torch.maximum(diffuse,  diffuse_rev)

    colors = (ambient + diffuse) * texels + specular
    return colors, pixel_coords_in_camera
