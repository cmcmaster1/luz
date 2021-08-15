# Learning rate annealing for lr_finder
lr_anneal <- torch::lr_scheduler(
  "lr_lambda",
  initialize = function(
    optimizer,
    start_lr = 1e-7,
    end_lr = 1e-1,
    n_iters = 100,
    last_epoch=-1,
    verbose=FALSE) {

    self$optimizer <- optimizer
    self$end_lr <- end_lr
    self$base_lrs <- start_lr
    self$iters <- n_iters

    super$initialize(optimizer, last_epoch, verbose)

  },
  get_lr = function() {
    if (self$last_epoch > 0) {
      lrs <- numeric(length(self$optimizer$param_groups))
      for (i in seq_along(self$optimizer$param_groups)) {
        lrs[i] <- self$optimizer$param_groups[[i]]$lr * (self$end_lr / self$optimizer$param_groups[[i]]$lr) ^ (self$last_epoch / self$iters)
      }
    } else {
      lrs <- as.numeric(self$base_lrs)
    }
    lrs
  }
)

# Early stopping callback for lr_finder
luz_callback_early_stopping_lr_finder <- luz_callback(
  name = "luz_callback_early_stopping_lr_finder",
  inherit = monitor_metrics,
  initialize = function(monitor = "train_loss", min_delta = 0, patience = 0,
                        mode="min", baseline=NULL) {

    super$initialize(monitor, mode, min_delta)

    self$patience <- patience
    self$baseline <- baseline

    if (!is.null(self$baseline))
      self$current_best <- baseline

    self$patience_counter <- 0L
  },

  on_epoch_end = function() {

    qty <- self$find_quantity()
    if (is.null(self$current_best))
      self$current_best <- qty

    if (ctx$get_metric("train", "AvgSmoothLoss", ctx$epoch) > 4*self$current_best) {
      rlang::signal("Early stopping", class = "early_stopping")
    }
  }
)

#' Internal metric that is used to track smoothed average loss for lr_finder
#' @noRd
luz_metric_loss_average_smooth <- luz_metric(
  abbrev = "AvgSmoothLoss",
  initialize = function(beta = 0.98) {
    self$values <- list()
    self$beta <- beta
  },
  update = function(preds, targets) {
    if (length(ctx$loss) == 1)
      loss <- ctx$loss[[1]]
    else
      loss <- ctx$loss

    self$values[[length(self$values) + 1]] <- loss
  },
  average_metric = function(x) {
    if (is.numeric(x[[1]]) || inherits(x[[1]], "torch_tensor"))
      x <- sapply(x, self$to_numeric)

    if (is.numeric(x)) {
      mean(x)
    } else if (is.list(x)) {
      lapply(purrr::transpose(x), self$average_metric)
    } else if (is.null(x)) {
      NULL
    } else {
      rlang::abort(c(
        "Average metric requires numeric tensor or values or list of them.")
      )
    }
  },
  compute = function() {
    torch::torch_lerp(self$average_metric(self$values), ctx$loss, self$beta)
  },
  to_numeric = function(x) {
    if (is.numeric(x))
      x
    else if (inherits(x, "torch_tensor"))
      as.numeric(x$to(device = "cpu"))
    else
      rlang::abort("Expected a numeric value or a tensor.")
  }
)


luz_callback_record_lr <- luz_callback(
  name = "luz_callback_profile_lr",
  on_train_batch_end = function() {
    loss <- ctx$loss$opt$cpu()$item()
    ctx$log("lr_finder", "lr", ctx$optimizers$opt$param_groups[[1]]$lr)
    ctx$log("lr_finder", "loss", loss)
  }
)


#' Learning Rate Finder
#'
#' @param object An nn_module that has been setup().
#' @param data (dataloader) A dataloader created with torch::dataloader()  used for learning rate finding.
#' @param steps (integer) The number of steps to iterate over in the learning rate finder. Default: 100.
#' @param start_lr (float) The smallest learning rate. Default: 1e-7.
#' @param end_lr (float) The highest learning rate. Default: 1e-1.
#' @param ... Other arguments passed to `fit`.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' ds <- torch::tensor_dataset(x = torch_randn(100, 10), y = torch_randn(100, 1))
#' dl <- torch::dataloader(ds, batch_size = 32)
#' model <- torch::nn_linear
#' model <- model %>% setup(
#'   loss = torch::nn_mse_loss(),
#'   optimizer = torch::optim_adam
#' ) %>%
#'   set_hparams(in_features = 10, out_features = 1)
#' records <- lr_finder(model, dl, verbose = FALSE)
#' plot(records)
#' }
#' @returns A dataframe with two columns: learning rate and loss
#' @export
lr_finder <- function(object, data, steps = 100, start_lr = 1e-7, end_lr = 1e-1, ...) {
  # adjust batch size so that the steps number adds to one batch
  new_bs <- floor(data$dataset$.length() / steps)
  data$batch_sampler$batch_size <- new_bs

  scheduler <- luz_callback_lr_scheduler(
    lr_anneal,
    verbose=FALSE,
    start_lr = start_lr,
    end_lr = end_lr,
    n_iters = steps,
    call_on="on_train_batch_begin"
  )

  lr_profiler <- luz_callback_record_lr()

  fitted <- object %>%
    set_opt_hparams(lr = start_lr) %>%
    fit(...,
        data = data,
        epochs = 1,
        callbacks = list(scheduler, lr_profiler),
    )

  lr_records <- data.frame(sapply(fitted$ctx$records$lr_finder, as.numeric))

  class(lr_records) <- c("lr_records", class(lr_records))
  lr_records
}

#' @export
print.lr_records <- function(x, ...) {
  NextMethod()
}

#' @export
plot.lr_records <- function(x, ...) {
  rlang::check_installed("ggplot2")
  x <- as.data.frame(x)
  ggplot2::ggplot(x, ggplot2::aes_string(x = "lr", y = "loss")) +
    ggplot2::geom_line() +
    ggplot2::scale_x_log10() +
    ggplot2::xlab("Learning Rate") +
    ggplot2::ylab("Loss")
}
